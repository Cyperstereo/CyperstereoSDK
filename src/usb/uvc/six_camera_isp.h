#ifndef CYPERSTEREO_SIX_CAMERA_ISP_H_
#define CYPERSTEREO_SIX_CAMERA_ISP_H_
#include "cyperstereo_api.h"

namespace cyperstereo {
// Six independent WB/exposure histories. A72 owners execute C1, C2,
// C3+C4 and C5+C6, respectively. No thread is created per frame.
class SixCameraIspProcessor {
 public:
  explicit SixCameraIspProcessor(
      IspPixelFormat output_format = IspPixelFormat::kBgr888)
      : output_format_(output_format) {
    CV_Assert(output_format == IspPixelFormat::kBgr888 ||
              output_format == IspPixelFormat::kUyvy422Bt601FullRange);
    try { for (int i=0; i<3; ++i) workers_[i] = std::thread([this,i] { Loop(i+1); }); }
    catch (...) { Stop(); throw; }
  }
  ~SixCameraIspProcessor() { Stop(); }
  SixCameraIspProcessor(const SixCameraIspProcessor&) = delete;
  SixCameraIspProcessor& operator=(const SixCameraIspProcessor&) = delete;

  // Read between Apply calls, just like IspProcessor::DigitalGain.
  double DigitalGain(int index) const {
    return index >= 0 && index < 6 ? wb_[index].DisplayExposure() : 1.0;
  }

  void Apply(const IspFrameJob *jobs) {
    CV_Assert(jobs);
    for (int i=0;i<6;++i) CV_Assert(jobs[i].raw && jobs[i].output && jobs[i].name);
    std::lock_guard<std::mutex> batch(batch_mutex_);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      jobs_=jobs; done_=0;
      for(auto& e:errors_) e=nullptr;
      ++generation_;
    }
    ready_.notify_all();
    {
      ScopedThreadAffinity pin(OwnerCpu(0));
      try { Process(0); } catch (...) { errors_[0]=std::current_exception(); }
    }
    std::unique_lock<std::mutex> lock(mutex_);
    finished_.wait(lock,[this] { return done_==3; });
    jobs_=nullptr;
    for (const auto& error:errors_) if (error) std::rethrow_exception(error);
  }

 private:
  static int OwnerCpu(int lane) {
#if defined(__linux__) && (defined(CYPERSTEREO_RK3576) || defined(CYPERSTEREO_RK3588))
    return 4+lane;
#else
    (void)lane; return -1;
#endif
  }
  void Stop() {
    { std::lock_guard<std::mutex> lock(mutex_); stop_=true; }
    ready_.notify_all();
    for (auto& t:workers_) if(t.joinable())t.join();
  }
  void Process(int index) {
    const auto& j=jobs_[index];
    // Only the two full-resolution cameras recruit small cores. Half images
    // remain on their A72 owners, leaving CPU2/3 available for capture.
#if defined(CYPERSTEREO_RK3576)
    FastIspBigLittleDefaultGuard defaults(index<2);
#endif
    FastIspBigLittleLaneGuard helper(index<2 ? index : -1);
    ApplyFastBalancedISPImpl(*j.raw,*j.output,wb_[index],j.name,j.sensor_gain,
        j.bayer,output_format_,j.raw_bayer_nr_done,j.metadata_region);
  }
  void Loop(int lane) {
    const int cpu=OwnerCpu(lane);
    if(cpu>=0) PinThreadToCpu(cpu);
    uint64_t seen=0;
    for(;;) {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock,[&] { return stop_ || generation_!=seen; });
        if(stop_)return;
        seen=generation_;
      }
      std::exception_ptr error;
      try {
        if(lane==1) Process(1);
        else if(lane==2) { Process(2); Process(3); }
        else { Process(4); Process(5); }
      } catch (...) { error=std::current_exception(); }
      { std::lock_guard<std::mutex> lock(mutex_); errors_[lane]=error; ++done_; }
      finished_.notify_one();
    }
  }
  const IspPixelFormat output_format_;
  std::array<WhiteBalance,6> wb_{};
  std::mutex batch_mutex_,mutex_;
  std::condition_variable ready_,finished_;
  const IspFrameJob* jobs_=nullptr;
  std::array<std::exception_ptr,4> errors_{};
  uint64_t generation_=0;
  int done_=0;
  bool stop_=false;
  std::array<std::thread,3> workers_;
};
} // namespace cyperstereo
#endif
