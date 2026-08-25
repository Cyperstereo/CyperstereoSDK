// Copyright 2018 Slightech Co., Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef CYPERSTEREO_UVC_V4L2_H_
#define CYPERSTEREO_UVC_V4L2_H_

#include "../uvc.h"
#include "../thread_priority.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <linux/usb/video.h>
#include <linux/usbdevice_fs.h>
#include <linux/uvcvideo.h>
#include <linux/videodev2.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>

CYPERSTEREO_BEGIN_NAMESPACE

namespace uvc {

// select() timeout (microseconds).  A larger window reduces busy-looping and
// avoids false "no data" judgments when the system is briefly loaded; 100ms
// keeps the polling thread responsive while not burning CPU.
#define UVC_V4L2_SELECT_TIMEOUT_US 100000

// How many consecutive select() timeouts (with no dequeued frame) are tolerated
// before declaring a stream timeout.  5 * 100ms = 0.5s: a healthy 30fps stream
// never goes silent for 500ms, so this cannot false-positive, and it cuts the
// outage from ~2.3s (old 2s window) to ~0.8s including the restart itself.
#define NO_DATA_MAX_COUNT 5
#define LIVING_MAX_COUNT 9000

// Number of V4L2 (kernel) capture buffers.  A deeper queue lets the kernel keep
// DMA'ing while the user thread is briefly preempted, and rides out short
// bursts of USB errors.  This is harmless headroom; it does NOT fix transfer
// errors (e.g. -EPROTO/-71) that originate at the USB/controller layer.
#define UVC_V4L2_BUFFER_COUNT 32


struct throw_error {
  throw_error() = default;

  explicit throw_error(const std::string &s) {
    ss << s;
  }

  ~throw_error() noexcept(false) {
    throw std::runtime_error(ss.str());
    // throw device_error(ss.str());
  }

  template<class T>
  throw_error &operator<<(const T &val) {
    ss << val;
    return *this;
  }

  std::ostringstream ss;
};

static int xioctl(int fh, int request, void *arg) {
  int r;
  do {
    r = ioctl(fh, request, arg);
  } while (r < 0 && errno == EINTR);
  return r;
}

// Some vendor ARM sysroots ship a C++11 libstdc++ without
// std::this_thread::sleep_for.  This file is Linux-only, so use the POSIX
// primitive directly and preserve the remaining delay if a signal interrupts
// the sleep.
static void sleep_for_milliseconds(unsigned int milliseconds) {
  timespec request;
  request.tv_sec = milliseconds / 1000U;
  request.tv_nsec = static_cast<long>(milliseconds % 1000U) * 1000000L;
  timespec remaining;
  while (nanosleep(&request, &remaining) < 0 && errno == EINTR) {
    request = remaining;
  }
}

// V4L2 sets DEVICE_CAPS on modern multi-node drivers and puts the capabilities
// of this particular /dev/videoX node in device_caps. Older/vendor kernels may
// not expose that flag (or even that struct member in their headers), so fall
// back to the legacy capabilities field without changing the selected ABI.
static uint32_t effective_device_caps(const v4l2_capability &cap) {
#ifdef V4L2_CAP_DEVICE_CAPS
  if (cap.capabilities & V4L2_CAP_DEVICE_CAPS)
    return cap.device_caps;
#endif
  return cap.capabilities;
}

// True for strings that look like a PCI BDF (e.g. "0000:00:14.0"). Some kernels
// expose the USB host controller's PCI slot as usbN/serial; that must never be
// treated as a camera serial number.
static bool looks_like_pci_bdf(const std::string &s) {
  // Minimal shape: hex:hex:hex.digit  (at least one ':' and a '.' near the end)
  const auto colon = s.find(':');
  const auto dot = s.rfind('.');
  if (colon == std::string::npos || dot == std::string::npos || dot + 1 >= s.size())
    return false;
  if (dot < colon) return false;
  for (char c : s) {
    if (!(std::isxdigit(static_cast<unsigned char>(c)) || c == ':' || c == '.'))
      return false;
  }
  return true;
}

// Read the USB device serial number from sysfs. V4L2 exposes the video node
// under /sys/class/video4linux/<name>/device (= USB interface); the parent is
// the USB device, which may carry a "serial" attribute. Do NOT walk further
// up to the USB host: that path's "serial" is the PCI BDF of the xHCI
// controller (observed as "0000:00:14.0" when FX3 iSerial is unset).
static std::string read_serial_number(const std::string &video_device) {
  const std::string path =
      "/sys/class/video4linux/" + video_device + "/device/../serial";
  std::ifstream serial_file(path);
  if (!serial_file) return "";
  std::string serial;
  std::getline(serial_file, serial);
  // Trim trailing whitespace / CR.
  while (!serial.empty() &&
         std::isspace(static_cast<unsigned char>(serial.back())))
    serial.pop_back();
  if (serial.empty() || looks_like_pci_bdf(serial)) return "";
  return serial;
}

struct buffer {
  void *start;
  size_t length;
};

struct context {
  context() {
    // VLOG(2) << __func__;
  }

  ~context() {
    // VLOG(2) << __func__;
  }
};

struct device {
  const std::shared_ptr<context> parent;

  std::string dev_name;  // Device name (typically of the form /dev/video*)

  std::string name;  // Device description name
  std::string serial_number;  // USB device serial number (if available)
  int vid, pid, mi;  // Vendor ID, product ID, and multiple interface index
  int fd = -1;       // File descriptor for this device

  int width, height, format, fps;
  video_channel_callback callback = nullptr;

  bool is_capturing = false;
  int last_start_error = 0;
  std::vector<buffer> buffers;

  std::thread thread;
  std::mutex device_mutex; // 新增互斥锁
  
  int no_data_count = 0;
  int living_count = 0;
  std::atomic<bool> stop{false};

  // --- stall diagnostics -----------------------------------------------------
  // Collected per-frame so that when a "stream time out" fires we can tell
  // WHERE the pipeline broke:
  //   - poll gap large  -> this (capture) thread was starved of CPU
  //                        (host overload / scheduling problem)
  //   - callback slow   -> consumer side too slow (SetStreamData / main-thread
  //                        mutex contention backing up into the poll thread)
  //   - both small      -> frames simply stopped arriving from the device
  //                        (USB link / FX3 / FPGA side; check dmesg for EPROTO)
  using diag_clock = std::chrono::steady_clock;
  diag_clock::time_point last_dqbuf_time{};    // last successful VIDIOC_DQBUF
  diag_clock::time_point last_poll_time{};     // previous poll() entry
  double max_poll_gap_ms = 0;    // worst poll-loop starvation since last report
  double max_cb_ms = 0;          // worst callback (consumer) time
  double max_frame_gap_ms = 0;   // worst inter-frame arrival gap
  uint64_t diag_frames = 0;

  // --- errored / short frame filter ------------------------------------------
  // With uvcvideo.nodrop=1 the kernel DELIVERS buffers whose USB transfer had
  // an error or ended short instead of silently dropping them.  Such a buffer
  // is only partially rewritten: its TAIL still holds the bytes of the frame
  // that occupied the same mmap slot one queue-cycle ago (UVC_V4L2_BUFFER_COUNT
  // = 32 buffers -> 32 frames = ~1067ms old).  The embedded metadata row is
  // the LAST row of the frame, i.e. exactly in that stale tail, so one bad
  // transfer shows up as a ~1.1s timestamp jump (33 frame periods) while the
  // stream itself never stops.  Detect via V4L2_BUF_FLAG_ERROR / bytesused and
  // skip the frame instead of parsing stale metadata.
  uint32_t full_frame_bytes = 0;   // steady-state bytesused of a good frame
  uint64_t bad_frame_count = 0;

  // Bad-frame STORM detector.  A single link error normally costs one short
  // frame.  But a link error can also knock the FX3's GPIF/DMA out of
  // alignment with its UVC framing, after which EVERY OTHER frame arrives
  // short (identical bytesused) indefinitely -- observed as 30fps dropping to
  // ~17fps with an endless [v4l2] BAD FRAME stream.  The device cannot heal
  // itself; restarting the video stream (STREAMOFF/ON) forces the firmware to
  // re-arm and realign.  Threshold: normal isolated events produce 1-3 bad
  // frames, a misalignment storm produces ~15/s, so 10 within 2s is decisive.
  diag_clock::time_point bad_window_start{};
  int bad_in_window = 0;
  // FAST-PATH misalignment detectors.  Two fingerprints of the
  // stuck-misaligned state, both far faster than the generic 10-frame
  // threshold (~550-600 ms at the observed ~15-18 bad/s alternating rate):
  //  1) Short frames with IDENTICAL bytesused repeating (stable offset,
  //     e.g. every other frame short by exactly 57312).  3 within the
  //     window is decisive (~200 ms).
  //  2) Bad-frame DENSITY: the misaligned offset can also drift, giving
  //     varying short lengths (observed 5054732 / 5186552 / 5185568 in one
  //     event, which fingerprint 1 misses), and EPROTO completions can mix
  //     zero-length ERROR buffers into the same burst (observed x86 event:
  //     3 shorts + zero-length errors took 549 ms to reach the generic
  //     10-frame threshold because zero-length was excluded here).
  //     Threshold history: 4 -> 6.  Firmware "solution five" now self-resyncs
  //     after 2 anomalous frames (~66 ms; was 3 until 2026-07),
  //     surfacing 2-3 bad frames here; a
  //     threshold of 4 would fire a redundant SECOND restart on top of the
  //     device's own recovery.  6 also stops penalizing transient URB-error
  //     bursts (AMD host: ~3 ERROR-flagged frames per burst, self-recovering
  //     -- a restart added 150 ms of outage for nothing).  A genuine wedge on
  //     old firmware still trips 6-in-window ~130 ms later than 4 did.
  // Zero-length restart artifacts never reach this detector: they are
  // filtered by the post-STREAMON grace period, so any zero-length buffer
  // seen here is a genuine mid-stream transfer error.  Deliberately NOT
  // reset by the good frames interleaved in the alternating pattern; reset
  // on window expiry/restart.
  uint32_t last_short_bytesused = 0;
  int identical_short_count = 0;
  int short_in_window = 0;
  // Time of the last successful STREAMON; used to identify the burst of
  // zero-length cancelled buffers our own restart produces (see the
  // restart-artifact grace period in poll()).
  diag_clock::time_point capture_start_time{};

  // --- stuck-fd / dead-driver recovery -----------------------------------------
  // After a severe EPROTO burst the uvcvideo driver can wedge ("Failed to
  // resubmit video URB (-1)" in dmesg): STREAMOFF/ON on the same fd no longer
  // brings frames back, and the fd sits in POLLERR state where select()
  // reports readable but DQBUF returns EAGAIN.  Without a guard that pair
  // hot-spins at SCHED_FIFO priority ("sched: RT throttling activated",
  // observed on Orange Pi 5 followed by a WaitForStream timeout abort).
  //  - eagain_streak: consecutive readable-but-EAGAIN wakes; each sleeps 1 ms
  //    to cap the spin, and a long streak (~300 ms solid) forces a restart.
  //  - restarts_since_frame: STREAMOFF/ON cycles without a single good frame
  //    in between; 3 in a row means the driver state is wedged and only
  //    closing and reopening the device node will clear it.
  int eagain_streak = 0;
  int restarts_since_frame = 0;
  diag_clock::time_point last_restart_time{};
  // Grows the retry interval while restarts keep failing to produce frames;
  // reset on the first good frame.  Bounds the worst-case buffer churn: an
  // unthrottled retry loop re-REQBUFS'ing 32 x 5 MB buffer sets against a
  // wedged device OOM-killed the whole board (vb2_vmalloc in the OOM stack).
  double restart_backoff_ms = 500.0;
  // Set when REQBUFS(0) fails: the kernel is still holding the old buffer
  // set (vb2 refuses to release while it thinks the queue is busy, e.g.
  // after a failed STREAMOFF on a wedged device).  Retrying REQBUFS(32) on
  // the same fd then allocates a FRESH 160 MB set each cycle while the old
  // ones linger -- the OOM mechanism above.  Only close() releases them
  // unconditionally, so escalate straight to reopen_device().
  bool kernel_bufs_stuck = false;

  // USB-level device reset (usbfs USBDEVFS_RESET) -- equivalent to a
  // physical replug.  Last-resort escalation: a severe EPROTO episode can
  // corrupt the FX3's control endpoint state so badly that it answers the
  // UVC probe/commit with garbage (observed: dwMaxVideoFrameSize ~252 MB,
  // byte pattern 0x0F0F0F11) and no amount of reopening the video node
  // helps.  Only a port reset makes the firmware reinitialize its USB
  // block.  Requires root (we already run under sudo for RT priorities).
  bool usb_reset_device() {
    const std::string video = dev_name.substr(5);  // "videoN"
    int busnum = -1, devnum = -1;
    std::ifstream("/sys/class/video4linux/" + video + "/device/../busnum")
        >> busnum;
    std::ifstream("/sys/class/video4linux/" + video + "/device/../devnum")
        >> devnum;
    if (busnum < 0 || devnum < 0) {
      std::cout << "[v4l2] usb reset: cannot resolve bus/dev for " << video
                << std::endl;
      return false;
    }
    char usbfs[64];
    snprintf(usbfs, sizeof(usbfs), "/dev/bus/usb/%03d/%03d", busnum, devnum);
    const int ufd = open(usbfs, O_WRONLY);
    if (ufd < 0) {
      std::cout << "[v4l2] usb reset: open " << usbfs << " failed: "
                << strerror(errno) << std::endl;
      return false;
    }
    const int rc = ioctl(ufd, USBDEVFS_RESET, 0);
    const int reset_errno = errno;
    close(ufd);
    // ENODEV after the ioctl means the device dropped off and re-enumerated
    // during the reset -- which is exactly the effect we wanted.
    const bool ok = (rc == 0) || (reset_errno == ENODEV);
    std::cout << "[v4l2] USB port reset of " << usbfs
              << (rc == 0 ? " done"
                          : (ok ? " done (device re-enumerated)" : " FAILED"))
              << std::endl;
    if (!ok)
      return false;
    // Wait for a matching video node to come back after re-enumeration (the
    // FX3 takes ~2 s on this kernel).  The node is often RENUMBERED
    // (video0 -> video2), so rediscover by VID/PID instead of waiting on
    // the old name.
    for (int i = 0; i < 60; ++i) {
      sleep_for_milliseconds(100);
      if (rediscover_dev_name())
        return true;
    }
    std::cout << "[v4l2] usb reset: no matching video node came back"
              << std::endl;
    return false;
  }

  // Scan /sys/class/video4linux for a video-capture node whose modalias
  // matches this device's VID/PID; update dev_name if found.  Needed after
  // a disconnect/reset because udev renumbers the nodes.
  bool rediscover_dev_name() {
    DIR *dir = opendir("/sys/class/video4linux");
    if (!dir)
      return false;
    char want[32];
    snprintf(want, sizeof(want), "usb:v%04Xp%04X", vid, pid);
    bool found = false;
    while (dirent *entry = readdir(dir)) {
      const std::string name = entry->d_name;
      if (name.substr(0, 5) != "video")
        continue;
      std::string modalias;
      if (!(std::ifstream("/sys/class/video4linux/" + name +
                          "/device/modalias") >> modalias))
        continue;
      if (modalias.compare(0, strlen(want), want) != 0)
        continue;
      // Two nodes match (video capture + metadata); take the one that
      // reports the video-capture capability.
      const std::string cand = "/dev/" + name;
      const int tfd = open(cand.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC, 0);
      if (tfd < 0)
        continue;
      v4l2_capability cap = {};
      const bool is_capture =
          ioctl(tfd, VIDIOC_QUERYCAP, &cap) == 0 &&
          (effective_device_caps(cap) & V4L2_CAP_VIDEO_CAPTURE);
      close(tfd);
      if (!is_capture)
        continue;
      if (cand != dev_name) {
        std::cout << "[v4l2] device node moved: " << dev_name << " -> "
                  << cand << std::endl;
        dev_name = cand;
      }
      found = true;
      break;
    }
    closedir(dir);
    return found;
  }

  // Consecutive reopen_device() cycles that still produced no frame; two in
  // a row escalates to the USB port reset above.  Reset on good frames.
  int reopens_since_frame = 0;

  void reopen_device() {
    std::lock_guard<std::mutex> lock(device_mutex);
    std::cout << "[v4l2] driver appears wedged (repeated restarts without "
                 "frames), reopening " << dev_name << std::endl;
    if (fd != -1) {
      close(fd);   // releases any kernel buffer set vb2 refused to free
      fd = -1;
    }
    kernel_bufs_stuck = false;

    if (++reopens_since_frame >= 2) {
      reopens_since_frame = 0;
      usb_reset_device();
    }

    // Give uvcvideo a moment to release the old file-handle state.
    sleep_for_milliseconds(200);
    fd = open(dev_name.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC, 0);
    if (fd < 0 && errno == ENOENT) {
      // Node gone: the device disconnected and re-enumerated under a new
      // number (e.g. /dev/video0 -> /dev/video2).  Look it up by VID/PID
      // instead of burning a whole retry cycle waiting for the port reset
      // escalation to do the same rediscovery.
      if (rediscover_dev_name())
        fd = open(dev_name.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC, 0);
    }
    if (fd < 0) {
      fd = -1;
      std::cout << "[v4l2] reopen failed: " << strerror(errno)
                << " (will retry on next watchdog cycle)" << std::endl;
    }
  }

  // All stream restarts funnel through here so the wedged-driver escalation
  // sees every attempt regardless of which detector triggered it.
  void restart_stream() {
    // Rate limit with exponential backoff: each attempt sets up 32 x 5 MB
    // of kernel buffers; on a wedged device an unthrottled retry loop
    // churns that setup several times per second, ballooning kernel memory
    // until the OOM killer takes the whole board down (observed on Orange
    // Pi 5, vb2_vmalloc in the OOM stack).  500 ms -> 1 s -> 2 s -> ... ->
    // 8 s cap while failing; reset to 500 ms by the first good frame.
    const auto now = diag_clock::now();
    if (last_restart_time.time_since_epoch().count() != 0 &&
        ms_between(last_restart_time, now) < restart_backoff_ms)
      return;
    last_restart_time = now;
    restart_backoff_ms = std::min(restart_backoff_ms * 2.0, 8000.0);

    stop_capture();
    ++restarts_since_frame;
    if (fd == -1 || kernel_bufs_stuck || restarts_since_frame >= 3) {
      restarts_since_frame = 0;
      reopen_device();
      if (fd == -1)
        return;
    }
    if (!start_capture())
      std::cout << "[v4l2] stream restart failed, will retry" << std::endl;
  }
  // Log throttling: during a storm (e.g. ~200 zero-length cancelled buffers
  // at stream start) printing every bad frame floods the console.  Print
  // details for the first few per window, silently count the rest, and emit
  // one aggregated summary line once enough have accumulated.
  uint64_t bad_suppressed = 0;

  // --- FX3 link-error telemetry (firmware "solution three") -------------------
  // The FX3 firmware exposes its error counters through UVC Extension Unit 3,
  // selector 2.  Wire format (LE uint32 each, running totals since boot):
  //   32 bytes (2026-07 firmware+):
  //     [0] usb3 phy_err  [1] usb3 lnk_err  [2] ep_underrun
  //     [3] dma_reset TOTAL
  //     [4] ..caused by frame watchdog     (producer stall / drain stall)
  //     [5] ..caused by commit failure     (USB consumer stalled)
  //     [6] ..caused by frame-size anomaly (FPGA->FX3 data corruption)
  //     [7] DMA buffer pool free bytes (static snapshot, not a counter)
  //   16 bytes (older firmware): fields [0..3] only.
  // The uvcvideo driver validates the query size against the device's
  // GET_LEN (mismatch = ENOBUFS without reaching the device), so probing
  // 32 then 16 auto-selects the right format.  Polled every 30 s and printed
  // as [fx3-stats] with per-window deltas; the per-cause split attributes
  // each device self-recovery to "host stall" vs "FPGA corruption" without
  // a UART hookup.  Firmware without the control at all fails both probes,
  // after which polling is disabled for the session.
  static constexpr uint8_t kFx3XuUnit = 3;
  static constexpr uint8_t kFx3XuLinkErrorStatsSelector = 2;
  static constexpr double kFx3StatsPollPeriodMs = 30000.0;
  bool fx3_stats_supported = true;
  uint16_t fx3_stats_len = 32;      // negotiated: 32 (new fw) or 16 (old fw)
  bool fx3_stats_len_probed = false;
  diag_clock::time_point last_fx3_stats_poll{};
  uint32_t fx3_stats_prev[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  bool fx3_stats_have_prev = false;

  static double ms_between(diag_clock::time_point a, diag_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  }

  void print_stall_diagnosis() {
    const auto now = diag_clock::now();
    const double since_last_frame =
        last_dqbuf_time.time_since_epoch().count() == 0
            ? -1.0 : ms_between(last_dqbuf_time, now);

    double load1 = -1.0;
    std::ifstream la("/proc/loadavg");
    if (la) la >> load1;

    std::cout << "[stall-diag] last_frame_age=" << since_last_frame << "ms"
              << "  max_poll_gap=" << max_poll_gap_ms << "ms"
              << "  max_callback=" << max_cb_ms << "ms"
              << "  max_frame_gap=" << max_frame_gap_ms << "ms"
              << "  frames_since_last_report=" << diag_frames
              << "  loadavg1m=" << load1 << std::endl;

    // One-line verdict. Thresholds: the poll loop normally iterates every
    // <=100ms (select timeout) and a 30fps frame arrives every ~33ms.
    if (max_poll_gap_ms > 300.0) {
      std::cout << "[stall-diag] verdict: capture thread was STARVED of CPU "
                   "(host overload) - check system load / RT priority" << std::endl;
    } else if (max_cb_ms > 100.0) {
      std::cout << "[stall-diag] verdict: consumer callback too SLOW "
                   "(main-thread/processing backlog stalled the poll thread)" << std::endl;
    } else {
      std::cout << "[stall-diag] verdict: host side healthy - device STOPPED "
                   "sending (USB link/FX3/FPGA; check `dmesg | grep -i uvc` "
                   "for EPROTO -71)" << std::endl;
    }
    max_poll_gap_ms = max_cb_ms = max_frame_gap_ms = 0;
    diag_frames = 0;
  }
  

  device(std::shared_ptr<context> parent, const std::string &name)
      : parent(parent), dev_name("/dev/" + name) {
    // VLOG(2) << __func__ << ": " << dev_name;

    struct stat st;
    if (stat(dev_name.c_str(), &st) < 0) {  // file status
      throw_error() << "Cannot identify '" << dev_name << "': " << errno << ", "
                    << strerror(errno);
    }
    if (!S_ISCHR(st.st_mode)) {  // character device?
      throw_error() << dev_name << " is no device";
    }

    if (!(std::ifstream("/sys/class/video4linux/" + name + "/name") >>
          this->name))
      throw_error() << "Failed to read name";

    std::string modalias;
    if (!(std::ifstream(
              "/sys/class/video4linux/" + name + "/device/modalias") >>
          modalias))
      throw_error() << "Failed to read modalias";
    if (modalias.size() < 14 || modalias.substr(0, 5) != "usb:v" ||
        modalias[9] != 'p')
      throw_error() << "Not a usb format modalias";
    if (!(std::istringstream(modalias.substr(5, 4)) >> std::hex >> vid))
      throw_error() << "Failed to read vendor ID";
    if (!(std::istringstream(modalias.substr(10, 4)) >> std::hex >> pid))
      throw_error() << "Failed to read product ID";
    if (!(std::ifstream(
              "/sys/class/video4linux/" + name + "/device/bInterfaceNumber") >>
          std::hex >> mi))
      throw_error() << "Failed to read interface number";

    serial_number = read_serial_number(name);

    fd = open(dev_name.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC, 0);
    if (fd < 0) {
      throw_error() << "Cannot open '" << dev_name << "': " << errno << ", "
                    << strerror(errno);
    }
    // A throwing constructor does not run ~device().  Keep the just-opened fd
    // guarded until all capability checks have completed so enumeration of an
    // unsupported node cannot leak a V4L2 handle into the capture process.
    struct ConstructorFdGuard {
      int *fd;
      ~ConstructorFdGuard() {
        if (fd != nullptr && *fd != -1) {
          close(*fd);
          *fd = -1;
        }
      }
      void release() { fd = nullptr; }
    } fd_guard{&fd};

    v4l2_capability cap;
    if (xioctl(fd, VIDIOC_QUERYCAP, &cap) < 0) {
      if (errno == EINVAL)
        throw_error() << dev_name << " is no V4L2 device";
      else
        throw_error() << "VIDIOC_QUERYCAP error " << errno << ", "
                      << strerror(errno);
    }
    const uint32_t caps = effective_device_caps(cap);
    if (!(caps & V4L2_CAP_VIDEO_CAPTURE))
      throw_error() << dev_name + " is no video capture device";
    if (!(caps & V4L2_CAP_STREAMING))
      throw_error() << dev_name + " does not support streaming I/O";

    // Select video input, video standard and tune here.
    v4l2_cropcap cropcap;
    cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd, VIDIOC_CROPCAP, &cropcap) == 0) {
      v4l2_crop crop;
      crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      crop.c = cropcap.defrect;  // reset to default
      if (xioctl(fd, VIDIOC_S_CROP, &crop) < 0) {
        switch (errno) {
          case EINVAL:
            break;  // Cropping not supported
          default:
            break;  // Errors ignored
        }
      }
    } else {
      throw_error() << dev_name + " is no video capture device";
    }  // Errors ignored
    fd_guard.release();
  }

  ~device() {
    // VLOG(2) << __func__;
    stop_streaming();
    no_data_count = 0;
    if (fd != -1 && close(fd) < 0) {
      std::cout << "close" << std::endl;
    }
  }

  bool pu_control_range(
      uint32_t id, int32_t *min, int32_t *max, int32_t *def) const {
    struct v4l2_queryctrl query;
    query.id = id;
    if (xioctl(fd, VIDIOC_QUERYCTRL, &query) < 0) {
      std::cout << "pu_control_range failed" << std::endl;
      return false;
    }
    if (min)
      *min = query.minimum;
    if (max)
      *max = query.maximum;
    if (def)
      *def = query.default_value;
    return true;
  }

  bool pu_control_query(uint32_t id, int query, int32_t *value) const {
    // CHECK_NOTNULL(value);
    struct v4l2_control control = {id, *value};
    if (xioctl(fd, query, &control) < 0) {
      std::cout << "pu_control_query failed" << std::endl;
      return false;
    }
    *value = control.value;
    return true;
  }

  bool xu_control_query(
      const xu &xu, uint8_t selector, uint8_t query, uint16_t size,
      uint8_t *data) const {
    // CHECK_NOTNULL(data);
    uvc_xu_control_query q = {xu.unit, selector, query, size, data};
    if (xioctl(fd, UVCIOC_CTRL_QUERY, &q) < 0) {
      std::cout << "xu_control_query failed" << std::endl;
      return false;
    }
    return true;
  }

  void set_format(
      int width, int height, int fourcc, int fps,
      video_channel_callback callback) {
    this->width = width;
    this->height = height;
    this->format = fourcc;
    this->fps = fps;
    this->callback = callback;
  }

  // Undo a partially-completed start_capture(): unmap what was mapped and
  // release the kernel buffers.  Safe to call with any intermediate state.
  void abort_capture_setup() {
    for (size_t i = 0; i < buffers.size(); i++) {
      if (buffers[i].start != nullptr && buffers[i].start != MAP_FAILED)
        munmap(buffers[i].start, buffers[i].length);
      buffers[i].start = nullptr;
    }
    buffers.clear();
    v4l2_requestbuffers req = {};
    req.count = 0;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
      std::cout << "[v4l2] WARN kernel refused to release buffers ("
                << strerror(errno) << "), will reopen device" << std::endl;
      kernel_bufs_stuck = true;
    }
  }

  // Returns true only if the stream is genuinely running with a fully valid
  // buffer set.  On ANY setup failure it rolls back and returns false --
  // previously it pressed on and set is_capturing=true regardless, which on
  // a wedged device left poll() dereferencing unmapped/MAP_FAILED buffers
  // (segfault) while the retry churn of half-built 32 x 5 MB buffer sets
  // drove the kernel into the OOM killer (both observed on Orange Pi 5).
  bool start_capture(bool suppress_busy_log = false) {
    last_start_error = 0;
    if (is_capturing) {
      std::cout << "Start capture failed, is capturing already" << std::endl;
      return true;
    }
    if (fd == -1)
      return false;

    v4l2_format fmt = {};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = format;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    if (xioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
      last_start_error = errno;
      if (!suppress_busy_log || last_start_error != EBUSY) {
        std::cout << "VIDIOC_S_FMT failed: " << strerror(last_start_error)
                  << std::endl;
      }
      return false;
    }
    // Validate what the driver actually negotiated.  A wedged device can
    // answer the UVC probe/commit with garbage, and S_FMT "succeeds" with
    // a corrupt frame size; REQBUFS below then multiplies that by 32
    // buffers in one vmalloc burst -- observed on Orange Pi 5 as the OOM
    // killer taking out the whole board seconds after start.  YUYV is
    // 2 bytes/pixel, so anything beyond ~2x the expected size is garbage.
    {
      const uint32_t expected =
          static_cast<uint32_t>(width) * static_cast<uint32_t>(height) * 2;
      if (fmt.fmt.pix.width != static_cast<uint32_t>(width) ||
          fmt.fmt.pix.height != static_cast<uint32_t>(height) ||
          fmt.fmt.pix.sizeimage < expected ||
          fmt.fmt.pix.sizeimage > expected * 2) {
        std::cout << "[v4l2] S_FMT negotiated bogus format: "
                  << fmt.fmt.pix.width << "x" << fmt.fmt.pix.height
                  << " sizeimage=" << fmt.fmt.pix.sizeimage
                  << " (expected " << width << "x" << height
                  << " ~" << expected << " bytes) -> device wedged, "
                     "aborting this start attempt" << std::endl;
        return false;
      }
    }

    v4l2_streamparm parm = {};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd, VIDIOC_G_PARM, &parm) < 0)
      std::cout << "VIDIOC_G_PARM " << std::endl;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = fps;
    if (xioctl(fd, VIDIOC_S_PARM, &parm) < 0)
      std::cout << "VIDIOC_S_PARM" << std::endl;

    // Init memory mapped IO
    v4l2_requestbuffers req = {};
    req.count = UVC_V4L2_BUFFER_COUNT;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
      last_start_error = errno;
      if (errno == EINVAL)
        std::cout << "does not support memory mapping " << std::endl;
      else
        std::cout << "VIDIOC_REQBUFS failed: " << strerror(errno) << std::endl;
      return false;
    }
    if (req.count < 2) {
      std::cout << "Insufficient buffer memory" << std::endl;
      abort_capture_setup();
      return false;
    }

    buffers.resize(req.count);
    for (size_t i = 0; i < buffers.size(); ++i) {
      buffers[i].start = nullptr;
      buffers[i].length = 0;
    }
    for (size_t i = 0; i < buffers.size(); ++i) {
      v4l2_buffer buf = {};
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;
      buf.index = i;
      if (xioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) {
        std::cout << "VIDIOC_QUERYBUF failed: " << strerror(errno) << std::endl;
        abort_capture_setup();
        return false;
      }
      buffers[i].length = buf.length;
      buffers[i].start = mmap(
          NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
          buf.m.offset);
      if (buffers[i].start == MAP_FAILED) {
        std::cout << "mmap failed: " << strerror(errno) << std::endl;
        buffers[i].start = nullptr;
        abort_capture_setup();
        return false;
      }
    }

    // Start capturing
    for (size_t i = 0; i < buffers.size(); ++i) {
      v4l2_buffer buf = {};
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;
      buf.index = i;
      if (xioctl(fd, VIDIOC_QBUF, &buf) < 0) {
        std::cout << "VIDIOC_QBUF failed: " << strerror(errno) << std::endl;
        abort_capture_setup();
        return false;
      }
    }

    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    for (int i = 0; i < 10; ++i) {
      if (xioctl(fd, VIDIOC_STREAMON, &type) < 0) {
        last_start_error = errno;
        sleep_for_milliseconds(100);
      } else {
        capture_start_time = diag_clock::now();
        is_capturing = true;
        return true;
      }
    }
    std::cout << "VIDIOC_STREAMON failed: " << strerror(errno) << std::endl;
    abort_capture_setup();
    return false;
  }

  void stop_capture() {
    std::lock_guard<std::mutex> lock(device_mutex);
    if (!is_capturing)
      return;
    is_capturing = false;
    // Stop streamining
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd, VIDIOC_STREAMOFF, &type) < 0)
      std::cout << "VIDIOC_STREAMOFF" << std::endl;

    for (size_t i = 0; i < buffers.size(); i++) {
      if (buffers[i].start == nullptr || buffers[i].start == MAP_FAILED)
        continue;
      if (munmap(buffers[i].start, buffers[i].length) < 0)
        std::cout << "munmap" << std::endl;
      buffers[i].start = nullptr;
    }
    buffers.clear();

    // Close memory mapped IO
    struct v4l2_requestbuffers req = {};
    req.count = 0;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
      if (errno == EINVAL) {
        std::cout << dev_name << "does not support memory mapping" << std::endl;
      } else {
        // vb2 refused to drop the buffer set (wedged queue).  Flag it so
        // the next restart goes through reopen_device() instead of
        // stacking a fresh 160 MB allocation on top of the stuck one.
        std::cout << "[v4l2] WARN kernel refused to release buffers ("
                  << strerror(errno) << "), will reopen device" << std::endl;
        kernel_bufs_stuck = true;
      }
    }
  }

  void poll() {
    // Bail out early if the device has been closed (e.g. during a soft
    // restart triggered by stop_capture()/start_capture()).  Continuing to
    // select() on a stale/closed fd reports EBADF and would falsely inflate
    // no_data_count, producing spurious "stream time out" warnings during the
    // restart transition.
    if (fd == -1 || !is_capturing) {
      no_data_count = 0;
      // Not capturing: sleep so this loop cannot hot-spin at SCHED_FIFO
      // priority, then keep retrying recovery here (nothing else would).
      // Recovery is throttled by the SAME rate limiter for both branches
      // (restart_stream checks it internally): each attempt can allocate a
      // fresh 32 x 5 MB kernel buffer set while the kernel releases old
      // sets asynchronously, so an unthrottled loop out-allocates the
      // release worker and OOMs the board (observed on Orange Pi 5 after
      // a mid-stream device re-enumeration behind a USB hub).
      sleep_for_milliseconds(250);
      if (fd == -1) {
        const auto now = diag_clock::now();
        if (last_restart_time.time_since_epoch().count() != 0 &&
            ms_between(last_restart_time, now) < restart_backoff_ms)
          return;
        last_restart_time = now;
        restart_backoff_ms = std::min(restart_backoff_ms * 2.0, 8000.0);

        // Only try to reopen once a matching device node is actually back
        // (after a disconnect it takes udev a moment to recreate it, and
        // it usually comes back under a NEW number -- rediscover by
        // VID/PID rather than waiting for the old name).
        if (!rediscover_dev_name())
          return;
        reopen_device();
        if (fd != -1)
          start_capture();
      } else {
        restart_stream();
      }
      return;
    }

    // Poll-loop starvation detector: consecutive poll() entries are normally
    // <= ~100ms apart (the select timeout). A much larger gap means THIS
    // thread did not get CPU time -> host-side scheduling problem.
    {
      const auto now = diag_clock::now();
      if (last_poll_time.time_since_epoch().count() != 0) {
        const double gap = ms_between(last_poll_time, now);
        if (gap > max_poll_gap_ms) max_poll_gap_ms = gap;
        if (gap > 300.0) {
          std::cout << "[stall-diag] WARN poll loop starved for " << gap
                    << "ms (capture thread lost the CPU)" << std::endl;
        }
      }
      last_poll_time = now;
    }

    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(fd, &fds);

    struct timeval tv = {0, UVC_V4L2_SELECT_TIMEOUT_US};

    if (select(fd + 1, &fds, NULL, NULL, &tv) < 0) {
      if (errno == EINTR)
        return;
      // fd may have just been closed by stop_capture(); don't treat this as a
      // stream stall, just exit this iteration quietly.
      if (errno == EBADF)
        return;
      std::cout << "select" << std::endl;
    }

    if (FD_ISSET(fd, &fds)) {
      v4l2_buffer buf;
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;
      if (xioctl(fd, VIDIOC_DQBUF, &buf) < 0) {
        if (errno == EAGAIN) {
          // select() said readable but there is no buffer.  Normally a rare
          // spurious wake; but with the fd in POLLERR state (wedged driver)
          // this repeats immediately and would hot-spin at SCHED_FIFO
          // priority.  Cap the loop rate and escalate a solid streak.
          ++eagain_streak;
          sleep_for_milliseconds(1);
          if (eagain_streak >= 300) {
            eagain_streak = 0;
            std::cout << "[v4l2] fd stuck readable-but-empty for ~300 ms "
                         "(POLLERR spin) -> restarting stream" << std::endl;
            restart_stream();
          }
          return;
        }
        if (errno == ENODEV) {
          // The device fell off the bus (USB3 link death or re-enumeration;
          // the kernel disconnected it).  Every further ioctl on this fd can
          // only return ENODEV too, so counting watchdog ticks or trying
          // STREAMOFF first just adds ~1 s of dead time (observed x86 link
          // death: 5 DQBUF ENODEV spins + 500 ms watchdog before recovery
          // started).  Escalate straight to the reopen / rediscover /
          // port-reset ladder.  Sleep because a dead fd keeps select()
          // returning readable instantly, and restart_stream() returns
          // without blocking while inside its retry backoff window.
          std::cout << "[v4l2] device disconnected (DQBUF ENODEV) -> "
                       "immediate reopen/rediscover" << std::endl;
          no_data_count = 0;
          kernel_bufs_stuck = true;  // REQBUFS(0) on a dead fd cannot work
          sleep_for_milliseconds(50);
          restart_stream();
          return;
        }
        std::cout << "VIDIOC_DQBUF failed: " << strerror(errno) << std::endl;
        // buf is NOT valid here; falling through would process garbage
        // (and requeue a bogus buffer index). Count it against the stream
        // watchdog instead.
        no_data_count++;
        if (no_data_count > NO_DATA_MAX_COUNT) {
          no_data_count = 0;
          living_count = 0;
          std::cout << " failed: v4l2 get stream time out, Try to reboot!" << std::endl;
          print_stall_diagnosis();
          restart_stream();
        }
        return;
      }

      // Frame arrived: update inter-frame gap stats (healthy 30fps ~= 33ms).
      eagain_streak = 0;
      {
        const auto now = diag_clock::now();
        if (last_dqbuf_time.time_since_epoch().count() != 0) {
          const double gap = ms_between(last_dqbuf_time, now);
          if (gap > max_frame_gap_ms) max_frame_gap_ms = gap;
        }
        last_dqbuf_time = now;
        ++diag_frames;
      }

      // Errored / short transfer? (delivered because of uvcvideo.nodrop=1)
      // Its tail -- including the metadata row -- is stale (~32 frames old),
      // so requeue it immediately and do NOT hand it to the consumer.
      {
        const bool flag_error = (buf.flags & V4L2_BUF_FLAG_ERROR) != 0;
        const bool short_frame =
            full_frame_bytes > 0 && buf.bytesused < full_frame_bytes;

        // Restart artifacts: our own STREAMOFF/ON (watchdog or storm
        // recovery) cancels all queued buffers, which then come back as a
        // burst of zero-length ERROR buffers microseconds after STREAMON.
        // They carry no data loss, but if fed to the storm detector they
        // trip the generic 10-frame threshold instantly and trigger a
        // SECOND restart of the just-restarted stream (observed on Orange
        // Pi 5: 600 ms watchdog restart followed back-to-back by a
        // "10 bad frames in 0.021 ms" storm restart, stretching the total
        // gap to 800 ms).  Discard them quietly during a grace period
        // after STREAMON.  A genuine post-restart error storm (the old
        // -71 EPROTO loop) keeps producing them past the grace period and
        // still trips the detector.
        if (flag_error && buf.bytesused == 0 &&
            capture_start_time.time_since_epoch().count() != 0 &&
            ms_between(capture_start_time, diag_clock::now()) < 1000.0) {
          if (xioctl(fd, VIDIOC_QBUF, &buf) < 0)
            std::cout << "VIDIOC_QBUF (restart-artifact requeue)" << std::endl;
          no_data_count = 0;
          return;
        }

        if (flag_error || short_frame) {
          ++bad_frame_count;
          // Only print details for the first few bad frames per window;
          // storms of hundreds of cancelled buffers get one summary later.
          if (bad_in_window < 3) {
            std::cout << "[v4l2] BAD FRAME skipped: flags=0x" << std::hex
                      << buf.flags << std::dec << (flag_error ? " (ERROR)" : "")
                      << "  bytesused=" << buf.bytesused << "/"
                      << full_frame_bytes
                      << "  seq=" << buf.sequence
                      << "  total_bad=" << bad_frame_count
                      << "  (metadata tail would be ~32 frames stale)"
                      << std::endl;
          } else {
            ++bad_suppressed;
          }
          if (xioctl(fd, VIDIOC_QBUF, &buf) < 0)
            std::cout << "VIDIOC_QBUF (bad-frame requeue)" << std::endl;
          no_data_count = 0;

          // Storm detection: many bad frames in a short window means the
          // device-side framing is stuck misaligned -> restart the stream.
          {
            const auto now = diag_clock::now();
            if (bad_in_window == 0 ||
                ms_between(bad_window_start, now) > 2000.0) {
              bad_window_start = now;
              bad_in_window = 0;
              identical_short_count = 0;
              last_short_bytesused = 0;
              short_in_window = 0;
            }
            ++bad_in_window;

            // Fast paths: see the detector comments at the member
            // declarations.  (1) identical-length short frames = stable
            // misalignment; (2) short-frame density = drifting misalignment.
            const char *fingerprint = nullptr;
            if (short_frame && buf.bytesused > 0) {
              if (buf.bytesused == last_short_bytesused) {
                if (++identical_short_count >= 3)
                  fingerprint = "3 identical-length short frames";
              } else {
                last_short_bytesused = buf.bytesused;
                identical_short_count = 1;
              }
            }
            // Density counts EVERY bad frame (short, zero-length or
            // error-flagged): restart artifacts were already filtered by the
            // grace period above, so this many within the window means the
            // link/framing is broken, not isolated noise.  (Threshold
            // rationale at the member declaration.)
            if (!fingerprint && ++short_in_window >= 6)
              fingerprint = "6 bad frames";

            if (fingerprint || bad_in_window >= 10) {
              std::cout << "[v4l2] BAD FRAME STORM ("
                        << (fingerprint ? fingerprint : "10 bad frames")
                        << " in " << ms_between(bad_window_start, now)
                        << " ms) -> device framing stuck misaligned, "
                           "restarting stream" << std::endl;
              bad_in_window = 0;
              identical_short_count = 0;
              last_short_bytesused = 0;
              short_in_window = 0;
              restart_stream();
            }
          }
          return;
        }

        // Good full-length frame: the stream is genuinely delivering again,
        // so clear the wedged-driver escalation counters and retry backoff.
        restarts_since_frame = 0;
        reopens_since_frame = 0;
        restart_backoff_ms = 500.0;
        if (buf.bytesused > full_frame_bytes)
          full_frame_bytes = buf.bytesused;

        // Aggregated summary of log-suppressed bad frames.  Only emit once a
        // few have accumulated (single suppressed frames between the good
        // frames of an alternating burst would otherwise spam one line each;
        // the running total_bad in the detail lines already carries the
        // count for those).
        if (bad_suppressed >= 5) {
          std::cout << "[v4l2] bad-frame burst: " << bad_suppressed
                    << " further bad frames suppressed from logging"
                    << "  (total_bad=" << bad_frame_count << ")" << std::endl;
          bad_suppressed = 0;
        }
      }

      if (callback) {
        const auto cb_start = diag_clock::now();
        callback(buffers[buf.index].start, [buf, this]() mutable {
          std::lock_guard<std::mutex> lock(device_mutex);
          if (is_capturing && fd != -1) {
            if (xioctl(fd, VIDIOC_QBUF, &buf) < 0) {
              // Never throw and never call stop_capture() here: throwing
              // from this continuation terminated the process on a wedged
              // device (QBUF fails EIO/EINVAL after a URB-resubmit failure),
              // and stop_capture() would self-deadlock on device_mutex.
              // Losing one buffer from the queue is fine -- the no-data
              // watchdog rebuilds the whole buffer set on restart.
              std::cout << "[v4l2] WARN buffer requeue failed: "
                        << strerror(errno) << std::endl;
            }
          }
        }
        );
        // Consumer-side cost of this frame (deinterleave + meta parse + any
        // wait on the main-thread mutex). If this exceeds the 33ms frame
        // budget the pipeline is falling behind on the host.
        const double cb_ms = ms_between(cb_start, diag_clock::now());
        if (cb_ms > max_cb_ms) max_cb_ms = cb_ms;
        if (cb_ms > 100.0) {
          std::cout << "[stall-diag] WARN consumer callback took " << cb_ms
                    << "ms (frame budget is ~33ms)" << std::endl;
        }
        if (living_count < LIVING_MAX_COUNT) {
          living_count++;
        } else {
          living_count = 0;
          std::cout << "UVC pulse detection,Please ignore." << std::endl;
        }
      }

      no_data_count = 0;
    } else {
      no_data_count++;
    }

    if (no_data_count > NO_DATA_MAX_COUNT) {
      no_data_count = 0;
      living_count = 0;
      std::cout << " failed: v4l2 get stream time out, Try to reboot!" << std::endl;
      print_stall_diagnosis();
      restart_stream();
    }
  }

  void poll_fx3_link_stats() {
    if (!fx3_stats_supported)
      return;

    const auto now = diag_clock::now();
    if (last_fx3_stats_poll.time_since_epoch().count() != 0 &&
        ms_between(last_fx3_stats_poll, now) < kFx3StatsPollPeriodMs)
      return;
    last_fx3_stats_poll = now;

    uint8_t data[32] = {0};
    {
      std::lock_guard<std::mutex> lock(device_mutex);
      if (fd == -1 || !is_capturing)
        return;
      const xu stats_xu{kFx3XuUnit, 0, {}};
      if (!xu_control_query(stats_xu, kFx3XuLinkErrorStatsSelector,
                            UVC_GET_CUR, fx3_stats_len, data)) {
        // First failure with the 32-byte format: probe the pre-2026-07
        // 16-byte format (the kernel rejects a size that mismatches the
        // device's GET_LEN, so old firmware fails the 32-byte query).
        if (!fx3_stats_len_probed && fx3_stats_len == 32 &&
            xu_control_query(stats_xu, kFx3XuLinkErrorStatsSelector,
                             UVC_GET_CUR, 16, data)) {
          fx3_stats_len = 16;
          std::cout << "[fx3-stats] 16-byte stats format (older firmware: "
                       "no per-cause reset counters)" << std::endl;
        } else {
          std::cout << "[fx3-stats] link-error stats XU not available "
                       "(older firmware?), disabling polling" << std::endl;
          fx3_stats_supported = false;
          return;
        }
      }
      fx3_stats_len_probed = true;
    }

    const int nvals = fx3_stats_len / 4;  // 8 (new fw) or 4 (old fw)
    uint32_t vals[8] = {0};
    for (int i = 0; i < nvals; ++i) {
      vals[i] = static_cast<uint32_t>(data[i * 4]) |
                (static_cast<uint32_t>(data[i * 4 + 1]) << 8) |
                (static_cast<uint32_t>(data[i * 4 + 2]) << 16) |
                (static_cast<uint32_t>(data[i * 4 + 3]) << 24);
    }

    std::cout << "[fx3-stats] usb3_phy_err=" << vals[0]
              << "  usb3_lnk_err=" << vals[1]
              << "  ep_underrun=" << vals[2]
              << "  dma_reset=" << vals[3];
    if (nvals >= 8) {
      std::cout << " (wd=" << vals[4] << " commit=" << vals[5]
                << " anomaly=" << vals[6] << ")"
                << "  pool_free=" << (vals[7] / 1024) << "KB";
    }
    if (fx3_stats_have_prev) {
      // Counters are running totals since FX3 boot and are never reset by
      // the SDK.  Any counter moving BACKWARD therefore means the FX3
      // rebooted (its globals reinitialized) -- don't print the huge
      // wrapped-unsigned deltas that would otherwise result.
      if (vals[0] < fx3_stats_prev[0] || vals[1] < fx3_stats_prev[1] ||
          vals[2] < fx3_stats_prev[2] || vals[3] < fx3_stats_prev[3]) {
        std::cout << "  (counters RESTARTED -> FX3/device rebooted since "
                     "last poll)";
      } else {
        std::cout << "  (delta/" << (kFx3StatsPollPeriodMs / 1000.0) << "s: +"
                  << (vals[0] - fx3_stats_prev[0]) << " +"
                  << (vals[1] - fx3_stats_prev[1]) << " +"
                  << (vals[2] - fx3_stats_prev[2]) << " +"
                  << (vals[3] - fx3_stats_prev[3]) << ")";
        if (vals[3] > fx3_stats_prev[3]) {
          // dma_reset increments = device-side in-stream self-recoveries.
          // With the FLAGB stream-gate firmware+FPGA each one costs only a
          // few whole frames.  The per-cause split (new firmware) says which
          // side to blame; without it, only the generic hint is printed.
          std::cout << "\n[fx3-stats] note: device self-resynced "
                    << (vals[3] - fx3_stats_prev[3])
                    << "x in this window";
          if (nvals >= 8) {
            const uint32_t wd = vals[4] - fx3_stats_prev[4];
            const uint32_t cf = vals[5] - fx3_stats_prev[5];
            const uint32_t an = vals[6] - fx3_stats_prev[6];
            std::cout << " [watchdog+" << wd << " commit_fail+" << cf
                      << " anomaly+" << an << "]";
            if (an > 0 && wd == 0 && cf == 0) {
              std::cout << " -> FPGA->FX3 data corruption self-healed "
                           "(thermal/GPIF side, NOT USB or host)";
            } else if (an == 0 && (wd > 0 || cf > 0)) {
              std::cout << " -> producer/consumer stall (host USB drain or "
                           "sensor stall), NOT data corruption";
            }
          } else {
            std::cout << " (device-internal recovery, not a USB/host "
                         "problem)";
          }
        }
      }
    } else {
      std::cout << "  (totals since device boot)";
    }
    std::cout << std::endl;

    for (int i = 0; i < nvals; ++i)
      fx3_stats_prev[i] = vals[i];
    fx3_stats_have_prev = true;
  }

  void start_streaming() {
    std::unique_lock<std::mutex> lock(device_mutex);
    if (!callback) {
      throw_error() << "[v4l2] cannot start " << dev_name
                    << ": video callback is empty";
    }

    if (thread.joinable() || is_capturing) {
      throw_error() << "[v4l2] streaming thread already running for "
                    << dev_name;
    }

    bool started = start_capture();
    if (!started && last_start_error == EBUSY) {
      // After the previous process closes its V4L2 fd, some uvcvideo/vb2
      // versions keep S_FMT busy for a short asynchronous teardown window.
      // Retry only here, before any recovery thread exists.  This also handles
      // a user restarting the sample immediately at the shell prompt without
      // confusing a transient release delay with a wedged camera.  A genuine
      // second owner remains bounded and fails below; it is never reopened or
      // USB-reset behind that owner's back.
      constexpr int kBusyRetryCount = 20;
      constexpr int kBusyRetryDelayMs = 100;
      std::cout << "[v4l2] " << dev_name
                << " is still being released; waiting up to "
                << kBusyRetryCount * kBusyRetryDelayMs << " ms" << std::endl;
      for (int attempt = 0;
           attempt < kBusyRetryCount && last_start_error == EBUSY;
           ++attempt) {
        sleep_for_milliseconds(kBusyRetryDelayMs);
        started = start_capture(true);
        if (started)
          break;
      }
      if (started) {
        std::cout << "[v4l2] " << dev_name
                  << " released; capture started" << std::endl;
      }
    }
    if (!started) {
      if (last_start_error == EBUSY) {
        throw_error()
            << "[v4l2] " << dev_name
            << " is busy; another process still owns the camera. Stop the old "
               "capture process (check `fuser -v "
            << dev_name << "`) and retry.";
      }
      throw_error() << "[v4l2] cannot start capture on " << dev_name
                    << (last_start_error == 0
                            ? std::string()
                            : std::string(": ") +
                                  strerror(last_start_error));
    }

    try {
      thread = std::thread([this]() {
        // Highest priority: a late dequeue means a permanently dropped frame.
        ApplyThreadPriority(ThreadRole::kPoll, "uvc-poll");
        while (!stop) {
          poll();
          poll_fx3_link_stats();
        }
      });
    } catch (...) {
      // stop_capture() takes device_mutex itself.
      lock.unlock();
      stop_capture();
      throw;
    }
  }

  void stop_streaming() {
    if (thread.joinable()) {
      stop = true;
      thread.join();
      stop = false;
    }
    // Also covers the narrow case where STREAMON succeeded but constructing
    // the polling std::thread threw: the kernel queue must still be released.
    stop_capture();
  }
};

std::shared_ptr<context> create_context() {
  return std::make_shared<context>();
}

std::vector<std::shared_ptr<device>> query_devices(
    std::shared_ptr<context> context) {
  std::vector<std::shared_ptr<device>> devices;

  DIR *dir = opendir("/sys/class/video4linux");
  if (!dir) {
    std::cout << "Cannot access /sys/class/video4linux" << std::endl;
    return devices;
  }
  while (dirent *entry = readdir(dir)) {
    std::string name = entry->d_name;
    if (name == "." || name == "..")
      continue;

    // Resolve a pathname to ignore virtual video devices
    std::string path = "/sys/class/video4linux/" + name;
    char buff[PATH_MAX];
    ssize_t len = ::readlink(path.c_str(), buff, sizeof(buff) - 1);
    if (len != -1) {
      buff[len] = '\0';
      std::string real_path = std::string(buff);
      if (real_path.find("virtual") != std::string::npos)
        continue;
    }

    try {
      auto one_device = std::make_shared<device>(context, name);
      devices.push_back(one_device);
    } catch (const std::exception &e) {
      std::cout << "Not a USB video device" << std::endl;
    }
  }
  closedir(dir);

  return devices;
}

std::string get_name(const device &device) {
  return device.name;
}

int get_vendor_id(const device &device) {
  return device.vid;
}

int get_product_id(const device &device) {
  return device.pid;
}

std::string get_video_name(const device &device) {
  return device.dev_name;
}

std::string get_serial_number(const device &device) {
  return device.serial_number;
}

bool has_frame_size(const device &device, int width, int height) {
  if (device.fd < 0 || width <= 0 || height <= 0) return false;
  for (int fi = 0;; ++fi) {
    v4l2_fmtdesc fmtdesc{};
    fmtdesc.index = static_cast<__u32>(fi);
    fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(device.fd, VIDIOC_ENUM_FMT, &fmtdesc) < 0) break;
    for (int si = 0;; ++si) {
      v4l2_frmsizeenum frmsize{};
      frmsize.index = static_cast<__u32>(si);
      frmsize.pixel_format = fmtdesc.pixelformat;
      if (xioctl(device.fd, VIDIOC_ENUM_FRAMESIZES, &frmsize) < 0) break;
      if (frmsize.type == V4L2_FRMSIZE_TYPE_DISCRETE) {
        if (static_cast<int>(frmsize.discrete.width) == width &&
            static_cast<int>(frmsize.discrete.height) == height)
          return true;
      } else if (frmsize.type == V4L2_FRMSIZE_TYPE_STEPWISE ||
                 frmsize.type == V4L2_FRMSIZE_TYPE_CONTINUOUS) {
        const int min_w = static_cast<int>(frmsize.stepwise.min_width);
        const int max_w = static_cast<int>(frmsize.stepwise.max_width);
        const int min_h = static_cast<int>(frmsize.stepwise.min_height);
        const int max_h = static_cast<int>(frmsize.stepwise.max_height);
        if (width >= min_w && width <= max_w && height >= min_h &&
            height <= max_h)
          return true;
      }
    }
  }
  return false;
}

static uint32_t get_cid(Option option) {
  switch (option) {
    case Option::GAIN:
      return V4L2_CID_GAIN;
    case Option::BRIGHTNESS:
      return V4L2_CID_BRIGHTNESS;
    case Option::CONTRAST:
      return V4L2_CID_CONTRAST;
    default:
      std::cout << "No v4l2 cid for" << std::endl;
  }
}

bool pu_control_range(
    const device &device, Option option, int32_t *min, int32_t *max,
    int32_t *def) {
  return device.pu_control_range(get_cid(option), min, max, def);
}

bool pu_control_query(
    const device &device, Option option, pu_query query, int32_t *value) {
  int code;
  switch (query) {
    case PU_QUERY_SET:
      code = VIDIOC_S_CTRL;
      break;
    case PU_QUERY_GET:
      code = VIDIOC_G_CTRL;
      break;
    default:
      std::cout << "pu_control_query request code is unaccepted" << std::endl;
      return false;
  }
  return device.pu_control_query(get_cid(option), code, value);
}

bool xu_control_range(
    const device &device, const xu &xu, uint8_t selector, uint8_t id,
    int32_t *min, int32_t *max, int32_t *def) {
  bool ret = true;

  std::uint8_t data[3]{static_cast<uint8_t>(id | 0x80), 0, 0};

  if (!xu_control_query(device, xu, selector, XU_QUERY_SET, 3, data)) {
    std::cout << "xu_control_range query failed" << std::endl;
    ret = false;
  }

  if (xu_control_query(device, xu, selector, XU_QUERY_MIN, 3, data)) {
    *min = (data[1] << 8) | (data[2]);
  } else {
    std::cout << "xu_control_range query min failed" << std::endl;
    ret = false;
  }
  if (xu_control_query(device, xu, selector, XU_QUERY_MAX, 3, data)) {
    *max = (data[1] << 8) | (data[2]);
  } else {
    std::cout << "xu_control_range query max failed" << std::endl;
    ret = false;
  }
  if (xu_control_query(device, xu, selector, XU_QUERY_DEF, 3, data)) {
    *def = (data[1] << 8) | (data[2]);
  } else {
    std::cout << "xu_control_range query def failed" << std::endl;
    ret = false;
  }
  return ret;
}

bool xu_control_query(
    const device &device, const xu &xu, uint8_t selector, xu_query query,
    uint16_t size, uint8_t *data) {
  uint8_t code;
  switch (query) {
    case XU_QUERY_SET:
      code = UVC_SET_CUR;
      break;
    case XU_QUERY_GET:
      code = UVC_GET_CUR;
      break;
    case XU_QUERY_MIN:
      code = UVC_GET_MIN;
      break;
    case XU_QUERY_MAX:
      code = UVC_GET_MAX;
      break;
    case XU_QUERY_DEF:
      code = UVC_GET_DEF;
      break;
    default:
      std::cout << "xu_control_query request code is unaccepted" << std::endl;
      return false;
  }
  return device.xu_control_query(xu, selector, code, size, data);
}

void set_device_mode(
    device &device, int width, int height, int fourcc, int fps,  // NOLINT
    video_channel_callback callback) {
  device.set_format(width, height, fourcc, fps, callback);
}

void start_streaming(device &device, int /*num_transfer_bufs*/) {  // NOLINT
  device.start_streaming();
}

void stop_streaming(device &device) {  // NOLINT
  device.stop_streaming();
}

}  // namespace uvc

CYPERSTEREO_END_NAMESPACE

// Video4Linux (V4L) driver-specific documentation
//   https://linuxtv.org/downloads/v4l-dvb-apis/v4l-drivers/index.html
#endif  // CYPERSTEREO_UVC_V4L2_H_
