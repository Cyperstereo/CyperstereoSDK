// Temporary live preview: MJPEG over HTTP, for watching the ISP output from a
// browser at http://<board>:<port>/. Linux only.
//
// The capture loop calls Wants() and, when it returns true, Offer(); Offer()
// only copies the frames. Conversion, scaling and JPEG encoding run on one
// thread restricted to CPU0-3 (the RK3576 Cortex-A53 cluster) so they stay
// off the cores the ISP uses.
#pragma once

#if defined(__linux__)

#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <sched.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

class MjpegPreview {
 public:
  // Called for /set?key=value with keys other than view/q/fps.
  using SetHandler = std::function<void(const std::string &, const std::string &)>;
  using UyvyToBgr = void (*)(const cv::Mat &, cv::Mat &);

  MjpegPreview(int port, int max_fps, UyvyToBgr to_bgr, SetHandler on_set)
      : max_fps_(max_fps > 0 ? max_fps : 10),
        to_bgr_(to_bgr),
        on_set_(std::move(on_set)) {
    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) return;
    const int one = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (bind(listen_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) <
            0 ||
        listen(listen_fd_, 8) < 0) {
      std::perror("[preview] bind/listen");
      close(listen_fd_);
      listen_fd_ = -1;
      return;
    }
    encoder_ = std::thread([this] { EncodeLoop(); });
    acceptor_ = std::thread([this] { AcceptLoop(); });
    std::printf("[preview] http://<board-ip>:%d/  (max %d fps)\n", port,
                max_fps_.load());
  }

  ~MjpegPreview() {
    stop_ = true;
    if (listen_fd_ >= 0) shutdown(listen_fd_, SHUT_RDWR);
    in_cv_.notify_all();
    out_cv_.notify_all();
    if (acceptor_.joinable()) acceptor_.join();
    if (encoder_.joinable()) encoder_.join();
    // Client threads are detached; socket timeouts bound how long they last.
    for (int i = 0; i < 400 && active_clients_.load() > 0; ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if (listen_fd_ >= 0) close(listen_fd_);
  }

  bool ok() const { return listen_fd_ >= 0; }

  // True when a browser is watching, the encoder is idle and the frame rate
  // limit allows another frame.
  bool Wants() const {
    if (viewers_.load() <= 0 || busy_.load()) return false;
    const auto now = std::chrono::steady_clock::now();
    return now - last_offer_ >=
           std::chrono::milliseconds(1000 / (std::max)(1, max_fps_.load()));
  }

  // `pair`: show images[0] and images[1] side by side at full resolution,
  // titled names[0] / names[1], regardless of the selected view.
  void Offer(const cv::Mat *const *images, int n, bool uyvy,
             const std::string &label, bool pair = false,
             std::vector<std::string> names = {}) {
    last_offer_ = std::chrono::steady_clock::now();
    const int view = view_.load();
    {
      std::lock_guard<std::mutex> lock(in_mutex_);
      in_.resize(n);
      for (int i = 0; i < n; ++i)
        if (pair || view == 0 || view == i + 1) images[i]->copyTo(in_[i]);
      in_uyvy_ = uyvy;
      in_label_ = label;
      in_view_ = view;
      in_pair_ = pair;
      in_names_ = std::move(names);
      busy_ = true;
    }
    in_cv_.notify_one();
  }

  // 0 = all cameras, k = camera k.
  int view() const { return view_.load(); }

  std::string status() const {
    std::lock_guard<std::mutex> lock(status_mutex_);
    return status_;
  }

 private:
  static void PinToLittleCores() {
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c = 0; c < 4; ++c) CPU_SET(c, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  }

  void EncodeLoop() {
    PinToLittleCores();
    std::vector<cv::Mat> frames;
    std::vector<std::string> names;
    cv::Mat bgr, bgr2, cell, canvas;
    std::vector<uchar> jpeg;
    for (;;) {
      bool uyvy, pair;
      int view;
      std::string label;
      {
        std::unique_lock<std::mutex> lock(in_mutex_);
        in_cv_.wait(lock, [&] { return stop_.load() || busy_.load(); });
        if (stop_) return;
        frames.swap(in_);
        uyvy = in_uyvy_;
        view = in_view_;
        label = in_label_;
        pair = in_pair_;
        names = in_names_;
      }
      const int n = static_cast<int>(frames.size());
      const auto as_bgr = [&](const cv::Mat &m) -> const cv::Mat & {
        if (!uyvy) return m;
        to_bgr_(m, bgr);
        return bgr;
      };
      if (pair && n >= 2) {
        const cv::Mat &left = uyvy ? (to_bgr_(frames[0], bgr2), bgr2)
                                   : frames[0];
        cv::hconcat(left, as_bgr(frames[1]), canvas);
        for (int i = 0; i < 2 && i < static_cast<int>(names.size()); ++i)
          cv::putText(canvas, names[i], {i * frames[0].cols + 10,
                                         canvas.rows - 14},
                      cv::FONT_HERSHEY_SIMPLEX, 0.9, {0, 255, 255}, 2,
                      cv::LINE_AA);
      } else if (view >= 1 && view <= n && !frames[view - 1].empty()) {
        as_bgr(frames[view - 1]).copyTo(canvas);
      } else if (n > 0) {
        // Half-size cells, two per row.
        const cv::Size c(frames[0].cols / 2, frames[0].rows / 2);
        const int cols = n > 1 ? 2 : 1, rows = (n + 1) / 2;
        canvas.create(c.height * rows, c.width * cols, CV_8UC3);
        canvas.setTo(cv::Scalar::all(0));
        for (int i = 0; i < n; ++i) {
          if (frames[i].empty()) continue;
          cv::resize(as_bgr(frames[i]), cell, c, 0, 0, cv::INTER_AREA);
          cell.copyTo(canvas(cv::Rect((i % 2) * c.width, (i / 2) * c.height,
                                      c.width, c.height)));
          cv::putText(canvas, "image" + std::to_string(i + 1),
                      {(i % 2) * c.width + 8, (i / 2) * c.height + c.height - 10},
                      cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 255, 255}, 1,
                      cv::LINE_AA);
        }
      }
      if (!canvas.empty()) {
        int base = 0;
        const cv::Size ts =
            cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &base);
        cv::rectangle(canvas, {0, 0, ts.width + 16, ts.height + base + 12},
                      cv::Scalar::all(0), cv::FILLED);
        cv::putText(canvas, label, {8, ts.height + 6}, cv::FONT_HERSHEY_SIMPLEX,
                    0.7, {0, 255, 0}, 2, cv::LINE_AA);
        cv::imencode(".jpg", canvas, jpeg,
                     {cv::IMWRITE_JPEG_QUALITY, quality_.load()});
        {
          std::lock_guard<std::mutex> lock(out_mutex_);
          out_ = std::make_shared<std::vector<uchar>>(jpeg);
          ++out_seq_;
        }
        out_cv_.notify_all();
        std::lock_guard<std::mutex> lock(status_mutex_);
        status_ = label;
      }
      {
        std::lock_guard<std::mutex> lock(in_mutex_);
        frames.swap(in_);
        busy_ = false;
      }
    }
  }

  void AcceptLoop() {
    PinToLittleCores();
    while (!stop_) {
      const int fd = accept(listen_fd_, nullptr, nullptr);
      if (fd < 0) {
        if (stop_) return;
        continue;
      }
      timeval tv{2, 0};
      setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
      setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
      const int one = 1;
      setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
      ++active_clients_;
      std::thread([this, fd] {
        PinToLittleCores();
        Serve(fd);
        close(fd);
        --active_clients_;
      }).detach();
    }
  }

  static bool SendAll(int fd, const void *data, size_t len) {
    const char *p = static_cast<const char *>(data);
    while (len > 0) {
      const ssize_t k = send(fd, p, len, MSG_NOSIGNAL);
      if (k <= 0) return false;
      p += k;
      len -= static_cast<size_t>(k);
    }
    return true;
  }

  static bool SendText(int fd, const char *type, const std::string &body) {
    const std::string head = std::string("HTTP/1.1 200 OK\r\nContent-Type: ") +
                             type + "\r\nContent-Length: " +
                             std::to_string(body.size()) +
                             "\r\nCache-Control: no-store\r\n"
                             "Connection: close\r\n\r\n";
    return SendAll(fd, head.data(), head.size()) &&
           SendAll(fd, body.data(), body.size());
  }

  void Serve(int fd) {
    std::string req;
    char buf[1024];
    while (req.find("\r\n\r\n") == std::string::npos && req.size() < 8192) {
      const ssize_t k = recv(fd, buf, sizeof(buf), 0);
      if (k <= 0) return;
      req.append(buf, static_cast<size_t>(k));
    }
    if (req.compare(0, 4, "GET ") != 0) return;
    const size_t end = req.find(' ', 4);
    const std::string target = req.substr(4, end - 4);
    const size_t q = target.find('?');
    const std::string path = target.substr(0, q);
    const std::string query = q == std::string::npos ? "" : target.substr(q + 1);
    if (path == "/") {
      SendText(fd, "text/html; charset=utf-8", Page());
    } else if (path == "/status") {
      SendText(fd, "text/plain; charset=utf-8", status());
    } else if (path == "/set") {
      size_t pos = 0;
      while (pos < query.size()) {
        size_t amp = query.find('&', pos);
        if (amp == std::string::npos) amp = query.size();
        const std::string kv = query.substr(pos, amp - pos);
        const size_t eq = kv.find('=');
        const std::string k = kv.substr(0, eq);
        const std::string v = eq == std::string::npos ? "" : kv.substr(eq + 1);
        if (k == "view")
          view_ = std::atoi(v.c_str());
        else if (k == "q")
          quality_ = (std::max)(10, (std::min)(100, std::atoi(v.c_str())));
        else if (k == "fps")
          max_fps_ = (std::max)(1, (std::min)(30, std::atoi(v.c_str())));
        else if (on_set_)
          on_set_(k, v);
        pos = amp + 1;
      }
      SendText(fd, "text/plain", "ok");
    } else if (path == "/stream") {
      Stream(fd);
    } else {
      const char kNotFound[] =
          "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n"
          "Connection: close\r\n\r\n";
      SendAll(fd, kNotFound, sizeof(kNotFound) - 1);
    }
  }

  void Stream(int fd) {
    const char kHead[] =
        "HTTP/1.1 200 OK\r\nCache-Control: no-store\r\nConnection: close\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
    if (!SendAll(fd, kHead, sizeof(kHead) - 1)) return;
    ++viewers_;
    uint64_t seen = 0;
    while (!stop_) {
      std::shared_ptr<std::vector<uchar>> jpeg;
      {
        std::unique_lock<std::mutex> lock(out_mutex_);
        out_cv_.wait_for(lock, std::chrono::seconds(1),
                         [&] { return stop_.load() || out_seq_ != seen; });
        if (stop_) break;
        if (out_seq_ == seen) continue;
        seen = out_seq_;
        jpeg = out_;
      }
      const std::string part = "--frame\r\nContent-Type: image/jpeg\r\n"
                               "Content-Length: " +
                               std::to_string(jpeg->size()) + "\r\n\r\n";
      if (!SendAll(fd, part.data(), part.size()) ||
          !SendAll(fd, jpeg->data(), jpeg->size()) || !SendAll(fd, "\r\n", 2))
        break;
    }
    --viewers_;
  }

  static std::string Page() {
    return R"HTML(<!doctype html><html><head><meta charset="utf-8">
<title>Cyperstereo ISP live preview</title>
<style>
body{background:#111;color:#ddd;font-family:sans-serif;margin:8px}
button{margin:2px 3px;padding:4px 10px}
#st{margin-left:12px;color:#8f8}
img{display:block;margin-top:6px;max-width:100%}
img.full{max-width:none}
</style></head><body>
<div>
ISP:
<button onclick="s('isp','hw')">HW ISP39</button>
<button onclick="s('isp','sw')">SW ISP</button>
<button onclick="s('isp','ab')" title="alternate frames, left HW / right SW">A/B</button>
<button onclick="s('save',1)" title="dump RAW + ISP jobs for offline replay">Save RAW</button>
&nbsp;View:
<button onclick="v(0)">2x2</button>
<button onclick="v(1)">image1</button>
<button onclick="v(2)">image2</button>
<button onclick="v(3)">image3</button>
<button onclick="v(4)">image4</button>
&nbsp;<label><input type="checkbox" id="full" onchange="f()">1:1</label>
&nbsp;JPEG q:
<button onclick="s('q',80)">80</button>
<button onclick="s('q',95)">95</button>
<span id="st"></span>
</div>
<img id="im" src="/stream">
<script>
function s(k,x){fetch('/set?'+k+'='+x);}
function v(i){s('view',i);}
function f(){document.getElementById('im').className=
  document.getElementById('full').checked?'full':'';}
setInterval(function(){fetch('/status').then(function(r){return r.text();})
  .then(function(t){document.getElementById('st').textContent=t;});},1000);
</script></body></html>
)HTML";
  }

  int listen_fd_ = -1;
  std::atomic<int> max_fps_;
  std::atomic<int> quality_{90};
  std::atomic<int> view_{0};
  std::atomic<int> viewers_{0};
  std::atomic<bool> busy_{false};
  std::atomic<bool> stop_{false};
  std::chrono::steady_clock::time_point last_offer_{};
  UyvyToBgr to_bgr_;
  SetHandler on_set_;

  std::mutex in_mutex_;
  std::condition_variable in_cv_;
  std::vector<cv::Mat> in_;
  bool in_uyvy_ = false;
  bool in_pair_ = false;
  int in_view_ = 0;
  std::string in_label_;
  std::vector<std::string> in_names_;

  std::mutex out_mutex_;
  std::condition_variable out_cv_;
  std::shared_ptr<std::vector<uchar>> out_;
  uint64_t out_seq_ = 0;

  mutable std::mutex status_mutex_;
  std::string status_;

  std::atomic<int> active_clients_{0};
  std::thread encoder_, acceptor_;
};

#endif  // __linux__
