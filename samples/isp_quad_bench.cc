// Offline replay of the four-camera fast-balanced ISP.
//
//   isp_quad_bench run <isp_jobs_N.txt> <out_prefix> [iterations] [par|seq]
//     Replays the RAW planes written by capture_image_imu --save-at (with
//     CYPERSTEREO_DUMP_ISP_JOBS=1 set) and
//     reports wall/CPU time per frame. The last frame is written as
//     <out_prefix>_camN.png.
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
    job.bayer = static_cast<BayerConversion>(bayer);
    job.nr_done = nr != 0;
    job.raw = cv::imread(dir + job.file, cv::IMREAD_GRAYSCALE);
    if (job.raw.empty()) {
      std::cerr << "cannot read " << dir + job.file << std::endl;
      return false;
    }
    jobs.push_back(std::move(job));
  }
  return jobs.size() == 4;
}

int Run(int argc, char **argv) {
  if (argc < 4) return 64;
  std::vector<ReplayJob> jobs;
  if (!LoadJobs(argv[2], jobs)) {
    std::cerr << "invalid job file " << argv[2] << std::endl;
    return 65;
  }
  const std::string prefix = argv[3];
  const int iterations = argc > 4 ? std::max(40, std::atoi(argv[4])) : 200;
  const bool sequential = argc > 5 && std::string(argv[5]) == "seq";
  constexpr int kWarmup = 30;
  static const char *const kNames[4] = {"cam1", "cam2", "cam3", "cam4"};

  IspProcessor parallel(IspMode::kFastBalancedBgr888);
  std::vector<std::unique_ptr<IspProcessor>> single;
  for (int i = 0; i < 4; ++i)
    single.emplace_back(new IspProcessor(IspMode::kFastBalancedBgr888));
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
                jobs[i].output);
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
  if (argc > 1 && std::string(argv[1]) == "compare") return Compare(argc, argv);
  std::cerr << "usage: isp_quad_bench run <isp_jobs.txt> <out_prefix> "
               "[iterations] [par|seq]\n"
               "       isp_quad_bench compare <reference.png> <candidate.png>"
            << std::endl;
  return 64;
}
