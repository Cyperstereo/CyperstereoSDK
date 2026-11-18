// Offline replay of the four-camera fast-balanced ISP.
//
//   isp_quad_bench run <isp_jobs_N.txt> <out_prefix> [iterations] [par|seq]
//                      [bgr|uyvy]
//     Replays the RAW planes written by capture_image_imu --save-at (with
//     CYPERSTEREO_DUMP_ISP_JOBS=1 set) and
//     reports wall/CPU time per frame. The last frame is written as
//     <out_prefix>_camN.png (UYVY output converted to BGR).
//   isp_quad_bench compare <reference.png> <candidate.png>
//     Prints max/mean absolute difference, PSNR and luma SSIM.
#include <time.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "../src/usb/uvc/cyperstereo_api.h"

using namespace cyperstereo;

namespace {

struct ReplayJob {
  std::string file;
  cv::Mat raw;
  cv::Mat output;
  double gain = 1.0;
  BayerConversion bayer = BayerConversion::kColorBayerRg2Bgr;
  bool nr_done = false;
  RawMetadataRegion region;
  double nr_scale = 1.0;
  bool half = false;
};

double NowMs(clockid_t clock) {
  timespec ts{};
  clock_gettime(clock, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

bool LoadJobs(const std::string &path, std::vector<ReplayJob> &jobs) {
  std::ifstream in(path);
  if (!in) return false;
  const size_t slash = path.find_last_of('/');
  const std::string dir = slash == std::string::npos ? "" : path.substr(0, slash + 1);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::istringstream fields(line);
    ReplayJob job;
    int bayer = 0, nr = 0;
    fields >> job.file >> job.gain >> bayer >> nr >> job.region.row >>
        job.region.first_col >> job.region.end_col;
    if (!fields) return false;
    int half = 0;
    if (fields >> job.nr_scale >> half) job.half = half != 0;
    job.bayer = static_cast<BayerConversion>(bayer);
    job.nr_done = nr != 0;
    job.raw = cv::imread(dir + job.file, cv::IMREAD_GRAYSCALE);
    if (job.raw.empty()) {
      std::cerr << "cannot read " << dir + job.file << std::endl;
      return false;
    }
    jobs.push_back(std::move(job));
  }
  return jobs.size() == 4 || jobs.size() == 6;
}

// BT.601 full range, the fast ISP's UYVY definition.
cv::Mat Uyvy422Bt601FullRangeToBgr(const cv::Mat &uyvy) {
  cv::Mat bgr(uyvy.rows, uyvy.cols, CV_8UC3);
  for (int y = 0; y < uyvy.rows; ++y) {
    const uchar *src = uyvy.ptr<uchar>(y);
    uchar *dst = bgr.ptr<uchar>(y);
    for (int x = 0; x + 1 < uyvy.cols; x += 2, src += 4, dst += 6) {
      const double cb = src[0] - 128.0, cr = src[2] - 128.0;
      for (int k = 0; k < 2; ++k) {
        const double luma = src[1 + 2 * k];
        dst[3 * k + 0] = cv::saturate_cast<uchar>(luma + 1.772 * cb);
        dst[3 * k + 1] =
            cv::saturate_cast<uchar>(luma - 0.344136 * cb - 0.714136 * cr);
        dst[3 * k + 2] = cv::saturate_cast<uchar>(luma + 1.402 * cr);
      }
    }
  }
  return bgr;
}

// Same-colour horizontal 2:1 bin. Phase is unchanged, so the source camera's
// BayerConversion still applies. 1280 -> 640, the SC136HGS six-image half
// width the FPGA produces before the host demosaics at full height.
void BayerHBin2SameColor(const cv::Mat &src, cv::Mat &dst) {
  CV_Assert(src.type() == CV_8U && src.cols >= 4 && (src.cols % 4) == 0);
  dst.create(src.rows, src.cols / 2, CV_8U);
  for (int y = 0; y < src.rows; ++y) {
    const uchar *s = src.ptr<uchar>(y);
    uchar *d = dst.ptr<uchar>(y);
    for (int x = 0; x < dst.cols; ++x) {
      const int a = ((x >> 1) << 2) + (x & 1);
      d[x] = static_cast<uchar>((s[a] + s[a + 2] + 1) >> 1);
    }
  }
}

// SC136HGS six-image schedule (not the AR0234 six-thread scheduler): the two
// full planes and the four demosaic-first halves, as one ApplyParallelWith
// call ("pair", default) or as two ApplyParallel calls ("split"). A six-line
// job file from capture_image_imu holds the real halves; with a quad file
// the halves are a same-colour horizontal bin of the four quad RAWs, the
// pixel count and Bayer phase of the real half lanes.
int RunSix(int argc, char **argv) {
  if (argc < 4) return 64;
  std::vector<ReplayJob> jobs;
  if (!LoadJobs(argv[2], jobs)) {
    std::cerr << "invalid job file " << argv[2] << std::endl;
    return 65;
  }
  const std::string prefix = argv[3];
  const int iterations = argc > 4 ? std::max(40, std::atoi(argv[4])) : 120;
  const bool split = argc > 5 && std::string(argv[5]) == "split";
  constexpr int kWarmup = 20;
  const bool real = jobs.size() == 6;
  cv::Mat half_raw[4];
  double half_gain[4], half_scale[4];
  BayerConversion half_bayer[4];
  for (int i = 0; i < 4; ++i) {
    const ReplayJob &src = jobs[real ? i + 2 : i];
    if (real)
      half_raw[i] = src.raw;
    else
      BayerHBin2SameColor(src.raw, half_raw[i]);
    half_gain[i] = src.gain;
    half_bayer[i] = src.bayer;
    half_scale[i] = real ? src.nr_scale : kHalfResTallNrGainScale;
  }
  cv::Mat half_out[4];
  const bool uyvy = argc > 6 && std::string(argv[6]) == "uyvy";
  const IspMode mode = uyvy ? IspMode::kFastBalancedUyvy422Bt601FullRange
                            : IspMode::kFastBalancedBgr888;
  IspProcessor full(mode);
  IspProcessor halves(mode);
  IspFrameJob full_jobs[] = {
      {jobs[0].raw, jobs[0].output, "cam1", jobs[0].gain, jobs[0].bayer,
       jobs[0].nr_done, jobs[0].region},
      {jobs[1].raw, jobs[1].output, "cam2", jobs[1].gain, jobs[1].bayer,
       jobs[1].nr_done, jobs[1].region},
  };
  static const char *const kHalf[4] = {"half0", "half1", "half2", "half3"};
  IspFrameJob half_jobs[4] = {
      {half_raw[0], half_out[0], kHalf[0], half_gain[0], half_bayer[0], false,
       {}, half_scale[0], true},
      {half_raw[1], half_out[1], kHalf[1], half_gain[1], half_bayer[1], false,
       {}, half_scale[1], true},
      {half_raw[2], half_out[2], kHalf[2], half_gain[2], half_bayer[2], false,
       {}, half_scale[2], true},
      {half_raw[3], half_out[3], kHalf[3], half_gain[3], half_bayer[3], false,
       {}, half_scale[3], true},
  };
  std::vector<double> wall, cpu, full_wall, half_wall;
  int full_hw = 0, half_hw = 0;
  for (int it = 0; it < iterations; ++it) {
    const double w0 = NowMs(CLOCK_MONOTONIC);
    const double c0 = NowMs(CLOCK_PROCESS_CPUTIME_ID);
    double mid = w0;
    if (split) {
      full.ApplyParallel(full_jobs, 2);
#if defined(CYPERSTEREO_RK_HW_ISP)
      full_hw = detail::RkHwIspLastBatchOnHardware() ? 1 : 0;
#endif
      mid = NowMs(CLOCK_MONOTONIC);
      halves.ApplyParallel(half_jobs, 4);
#if defined(CYPERSTEREO_RK_HW_ISP)
      half_hw = detail::RkHwIspLastBatchOnHardware() ? 1 : 0;
#endif
    } else {
      full.ApplyParallelWith(full_jobs, 2, halves, half_jobs, 4);
#if defined(CYPERSTEREO_RK_HW_ISP)
      full_hw = half_hw = detail::RkHwIspLastBatchOnHardware() ? 1 : 0;
#endif
    }
    if (it >= kWarmup) {
      const double end = NowMs(CLOCK_MONOTONIC);
      wall.push_back(end - w0);
      cpu.push_back(NowMs(CLOCK_PROCESS_CPUTIME_ID) - c0);
      full_wall.push_back(mid - w0);
      half_wall.push_back(end - mid);
    }
  }
  const auto avg = [](const std::vector<double> &v) {
    double s = 0;
    for (double x : v) s += x;
    return s / v.size();
  };
  std::cout << std::fixed << std::setprecision(2)
            << "[six] " << (real ? "real" : "binned") << " "
            << (split ? "split" : "pair") << " gain=" << jobs[0].gain << "/"
            << half_gain[0] << " frames=" << wall.size()
            << " full_hw=" << full_hw << " half_hw=" << half_hw
            << " wall_avg=" << avg(wall) << " cpu_avg=" << avg(cpu)
            << " full_wall=" << avg(full_wall)
            << " half_wall=" << avg(half_wall) << " ms" << std::endl;
  std::cout << "[six] full " << jobs[0].output.cols << "x" << jobs[0].output.rows
            << "  half " << half_out[0].cols << "x" << half_out[0].rows
            << "  bayer " << static_cast<int>(jobs[0].bayer)
            << static_cast<int>(jobs[1].bayer) << "/"
            << static_cast<int>(half_bayer[0]) << static_cast<int>(half_bayer[1])
            << static_cast<int>(half_bayer[2]) << static_cast<int>(half_bayer[3])
            << std::endl;
  for (int i = 0; i < 2; ++i)
    cv::imwrite(prefix + "_full" + std::to_string(i + 1) + ".png",
                uyvy ? Uyvy422Bt601FullRangeToBgr(jobs[i].output)
                     : jobs[i].output);
  for (int i = 0; i < 4; ++i)
    cv::imwrite(prefix + "_half" + std::to_string(i + 1) + ".png",
                uyvy ? Uyvy422Bt601FullRangeToBgr(half_out[i]) : half_out[i]);
  return 0;
}

int Run(int argc, char **argv) {
  if (argc < 4) return 64;
  std::vector<ReplayJob> jobs;
  if (!LoadJobs(argv[2], jobs) || jobs.size() != 4) {
    std::cerr << "invalid job file " << argv[2] << std::endl;
    return 65;
  }
  const std::string prefix = argv[3];
  const int iterations = argc > 4 ? std::max(40, std::atoi(argv[4])) : 200;
  const bool sequential = argc > 5 && std::string(argv[5]) == "seq";
  const bool uyvy = argc > 6 && std::string(argv[6]) == "uyvy";
  const IspMode mode = uyvy ? IspMode::kFastBalancedUyvy422Bt601FullRange
                            : IspMode::kFastBalancedBgr888;
  constexpr int kWarmup = 30;
  static const char *const kNames[4] = {"cam1", "cam2", "cam3", "cam4"};

  IspProcessor parallel(mode);
  std::vector<std::unique_ptr<IspProcessor>> single;
  for (int i = 0; i < 4; ++i) single.emplace_back(new IspProcessor(mode));
  std::vector<IspFrameJob> frame_jobs;
  for (int i = 0; i < 4; ++i)
    frame_jobs.emplace_back(jobs[i].raw, jobs[i].output, kNames[i],
                            jobs[i].gain, jobs[i].bayer, jobs[i].nr_done,
                            jobs[i].region);

  std::vector<double> wall, cpu;
  std::vector<std::vector<double>> per_camera(4);
  for (int it = 0; it < iterations; ++it) {
    const double w0 = NowMs(CLOCK_MONOTONIC);
    const double c0 = NowMs(CLOCK_PROCESS_CPUTIME_ID);
    if (sequential) {
      for (int i = 0; i < 4; ++i) {
        const double t0 = NowMs(CLOCK_MONOTONIC);
        single[i]->ApplyParallel(&frame_jobs[i], 1);
        if (it >= kWarmup) per_camera[i].push_back(NowMs(CLOCK_MONOTONIC) - t0);
      }
    } else {
      parallel.ApplyParallel(frame_jobs.data(), 4);
    }
    if (it >= kWarmup) {
      wall.push_back(NowMs(CLOCK_MONOTONIC) - w0);
      cpu.push_back(NowMs(CLOCK_PROCESS_CPUTIME_ID) - c0);
    }
  }
  const auto summarize = [](std::vector<double> v, double &avg, double &med,
                            double &p95) {
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (double x : v) sum += x;
    avg = sum / v.size();
    med = v[v.size() / 2];
    p95 = v[std::min(v.size() - 1, v.size() * 95 / 100)];
  };
  double avg, med, p95, cpu_avg, cpu_med, cpu_p95;
  summarize(wall, avg, med, p95);
  summarize(cpu, cpu_avg, cpu_med, cpu_p95);
  std::cout << std::fixed << std::setprecision(2) << "[bench] mode="
            << (sequential ? "seq" : "par") << " frames=" << wall.size()
            << " wall_avg=" << avg << " wall_med=" << med << " wall_p95=" << p95
            << " cpu_avg=" << cpu_avg << " ms/frame" << std::endl;
  if (sequential) {
    for (int i = 0; i < 4; ++i) {
      summarize(per_camera[i], avg, med, p95);
      std::cout << "[bench] " << kNames[i] << " avg=" << avg
                << " med=" << med << " p95=" << p95 << " ms" << std::endl;
    }
  }
  for (int i = 0; i < 4; ++i)
    cv::imwrite(prefix + "_cam" + std::to_string(i + 1) + ".png",
                uyvy ? Uyvy422Bt601FullRangeToBgr(jobs[i].output)
                     : jobs[i].output);
  return 0;
}

double LumaSsim(const cv::Mat &a_bgr, const cv::Mat &b_bgr) {
  cv::Mat a, b;
  cv::cvtColor(a_bgr, a, cv::COLOR_BGR2GRAY);
  cv::cvtColor(b_bgr, b, cv::COLOR_BGR2GRAY);
  a.convertTo(a, CV_32F);
  b.convertTo(b, CV_32F);
  const double c1 = 6.5025, c2 = 58.5225;
  const cv::Size window(11, 11);
  cv::Mat mu_a, mu_b, aa, bb, ab;
  cv::GaussianBlur(a, mu_a, window, 1.5);
  cv::GaussianBlur(b, mu_b, window, 1.5);
  cv::GaussianBlur(a.mul(a), aa, window, 1.5);
  cv::GaussianBlur(b.mul(b), bb, window, 1.5);
  cv::GaussianBlur(a.mul(b), ab, window, 1.5);
  const cv::Mat mu_aa = mu_a.mul(mu_a), mu_bb = mu_b.mul(mu_b),
                mu_ab = mu_a.mul(mu_b);
  const cv::Mat numerator = (2 * mu_ab + c1).mul(2 * (ab - mu_ab) + c2);
  const cv::Mat denominator =
      (mu_aa + mu_bb + c1).mul((aa - mu_aa) + (bb - mu_bb) + c2);
  cv::Mat ssim;
  cv::divide(numerator, denominator, ssim);
  return cv::mean(ssim)[0];
}

int Compare(int argc, char **argv) {
  if (argc < 4) return 64;
  const cv::Mat a = cv::imread(argv[2], cv::IMREAD_COLOR);
  const cv::Mat b = cv::imread(argv[3], cv::IMREAD_COLOR);
  if (a.empty() || b.empty() || a.size() != b.size()) {
    std::cerr << "cannot compare " << argv[2] << " " << argv[3] << std::endl;
    return 65;
  }
  cv::Mat diff;
  cv::absdiff(a, b, diff);
  double max_diff = 0;
  cv::minMaxLoc(diff.reshape(1), nullptr, &max_diff);
  const cv::Scalar mean = cv::mean(diff);
  const double mean_abs = (mean[0] + mean[1] + mean[2]) / 3.0;
  cv::Mat diff_f;
  diff.convertTo(diff_f, CV_32F);
  const double mse = cv::mean(diff_f.mul(diff_f)).dot(cv::Scalar(1, 1, 1)) / 3.0;
  const double psnr = mse > 0 ? 10.0 * std::log10(255.0 * 255.0 / mse) : 999.0;
  cv::Mat channel_max;
  cv::Mat planes[3];
  cv::split(diff, planes);
  cv::max(planes[0], planes[1], channel_max);
  cv::max(channel_max, planes[2], channel_max);
  const double pixels = static_cast<double>(channel_max.total());
  const double over2 = cv::countNonZero(channel_max > 2) * 100.0 / pixels;
  const double over8 = cv::countNonZero(channel_max > 8) * 100.0 / pixels;
  std::cout << std::fixed << std::setprecision(4) << "[compare] max=" << max_diff
            << " mean_abs=" << mean_abs << " psnr=" << std::setprecision(2)
            << psnr << "dB ssim=" << std::setprecision(5) << LumaSsim(a, b)
            << " px>2=" << std::setprecision(3) << over2 << "% px>8=" << over8
            << "%" << std::endl;
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  cv::setNumThreads(1);
  GetThreadPriorityConfig().enable = false;
  if (argc > 1 && std::string(argv[1]) == "run") return Run(argc, argv);
  if (argc > 1 && std::string(argv[1]) == "six") return RunSix(argc, argv);
  if (argc > 1 && std::string(argv[1]) == "compare") return Compare(argc, argv);
  std::cerr << "usage: isp_quad_bench run <isp_jobs.txt> <out_prefix> "
               "[iterations] [par|seq] [bgr|uyvy]\n"
               "       isp_quad_bench six <isp_jobs.txt> <out_prefix> "
               "[iterations] [pair|split] [bgr|uyvy]\n"
               "       isp_quad_bench compare <reference.png> <candidate.png>"
            << std::endl;
  return 64;
}
