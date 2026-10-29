#include "../src/usb/uvc/six_camera_isp.h"
#include <chrono>
#include <iostream>
#include <algorithm>
using namespace cyperstereo;
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
static std::vector<IspFrameJob> Jobs(cv::Mat* r,cv::Mat* o,double gain,bool nr=false,bool metadata=false){
 std::vector<IspFrameJob> j;
 static const char* names[]={"cam1","cam2","half-cam3","half-cam4","half-cam5","half-cam6"};
 for(int i=0;i<6;++i){RawMetadataRegion region; if(metadata){region.row=2;region.end_col=std::min(126,r[i].cols);}
  j.emplace_back(r[i],o[i],names[i],gain,i%2?BayerConversion::kColorBayerBg2Bgr:BayerConversion::kColorBayerRg2Bgr,nr,region);}
 return j;
}
static void Legacy(IspProcessor& a,IspProcessor& b,std::vector<IspFrameJob>& j){a.ApplyParallel(j.data(),2);b.ApplyParallel(j.data()+2,4);}
int main(int argc,char**argv){
 cv::setNumThreads(1);FastIspAr0234Profile().store(true);GetThreadPriorityConfig().enable=false;
 const bool uyvy=argc>1 && std::string(argv[1])=="verify-uyvy";
 const bool verify=uyvy || (argc>1 && std::string(argv[1])=="verify");
 const IspMode mode=uyvy?IspMode::kFastBalancedUyvy422Bt601FullRange:IspMode::kFastBalancedBgr888;
 const IspPixelFormat format=uyvy?IspPixelFormat::kUyvy422Bt601FullRange:IspPixelFormat::kBgr888;
 cv::Mat raw[6],out[6],ref[6];
 IspProcessor a(mode),b(mode);
 SixCameraIspProcessor six(format);
 if(verify){
  int frames=0; double maxdiff=0; cv::RNG rng(20261030);
  std::array<std::vector<uchar>,6> output_storage;
  // Independent scalar reference for the rounded same-CFA bin, including
  // odd half widths, padded ROIs and destinations aliasing an input.
  for(int width: {36,38,1280,1920,1922}) {
   cv::Mat parent[2],lanes[2],got[4],expected[4];
   for(int l=0;l<2;++l){parent[l].create(24,width+12,CV_8U);rng.fill(parent[l],cv::RNG::UNIFORM,0,256);lanes[l]=parent[l](cv::Rect(4,0,width,24));}
   for(int c=0;c<4;++c){expected[c].create(12,width/2,CV_8U);for(int y=0;y<12;++y)for(int x=0;x<width/2;++x){int iy=(y/2)*4+(y&1),ix=x+(c&1)*(width/2);expected[c].at<uchar>(y,x)=(int(lanes[c/2].at<uchar>(iy,ix))+lanes[c/2].at<uchar>(iy+2,ix)+1)>>1;}}
   SplitAndVBinHalfResLanes(lanes[0],lanes[1],got);
   for(int c=0;c<4;++c)if(cv::norm(got[c],expected[c],cv::NORM_INF))return 6;
   got[0]=lanes[0](cv::Rect(0,0,width/2,12));
   SplitAndVBinHalfResLanes(lanes[0],lanes[1],got);
   for(int c=0;c<4;++c)if(cv::norm(got[c],expected[c],cv::NORM_INF))return 7;
  }
  const int sizes[][2]={{1920,1080},{1280,812},{960,540},{1284,812},{36,8},{40,12},{1924,1084},{64,48}};
  const char* desired=std::getenv("CYPERSTEREO_FAST_BIG_LITTLE");std::string stages=desired?desired:"front,output";
  for(int s=0;s<8;++s){
   cv::Mat storage[6];
   for(int c=0;c<6;++c){int w=sizes[s][0],h=sizes[s][1];if(c>=2 && s<3){w=(w/2)&~3;h=(h/2)&~3;}
    storage[c].create(h,w+16,CV_8U);raw[c]=storage[c](cv::Rect(4,0,w,h));rng.fill(raw[c],cv::RNG::UNIFORM,0,256);}
   for(int n=0;n<8;++n){
    const double gains[]={1,2.99,3,4.92307692308,8,10,3.01,1};
    if(n==4)for(auto& r:raw)r.setTo(0);
    if(n==5)for(auto& r:raw)r.setTo(255);
    if(n==6)for(auto& r:raw)rng.fill(r,cv::RNG::UNIFORM,0,32);
    cv::Mat before[6];for(int c=0;c<6;++c)before[c]=raw[c].clone();
    // ROS wraps message-owned memory. Check that the ISP retains such
    // destinations and respects their row stride and surrounding guards.
    for(int c=0;c<6;++c) {
     const int channels=uyvy?2:3;
     const size_t stride=raw[c].cols*channels+32;
     output_storage[c].assign(raw[c].rows*stride+32,0xA5);
     out[c]=cv::Mat(raw[c].rows,raw[c].cols,uyvy?CV_8UC2:CV_8UC3,
                    output_storage[c].data()+16,stride);
    }
    auto original=Jobs(raw,ref,gains[n],n==7,true);auto optimized=Jobs(raw,out,gains[n],n==7,true);
    setenv("CYPERSTEREO_FAST_BIG_LITTLE","0",1);Legacy(a,b,original);
    setenv("CYPERSTEREO_FAST_BIG_LITTLE",stages.c_str(),1);six.Apply(optimized.data());
    for(int c=0;c<6;++c) {
     const size_t row_bytes=raw[c].cols*(uyvy?2:3),stride=row_bytes+32;
     const auto& bytes=output_storage[c];
     if(out[c].data!=bytes.data()+16)return 11;
     for(size_t k=0;k<16;++k)
      if(bytes[k]!=0xA5 || bytes[bytes.size()-1-k]!=0xA5)return 12;
     for(int y=0;y<raw[c].rows;++y)for(size_t x=row_bytes;x<stride;++x)
      if(bytes[16+y*stride+x]!=0xA5)return 13;
    }
    for(int c=0;c<6;++c)if(six.DigitalGain(c)!=(c<2?a.DigitalGain(c):b.DigitalGain(c-2)))return 10;
    for(int c=0;c<6;++c){double err=cv::norm(out[c],ref[c],cv::NORM_INF);maxdiff=std::max(maxdiff,err);++frames;if(err){std::cerr<<"MISMATCH "<<s<<' '<<n<<' '<<c<<' '<<err<<std::endl;return 2;}if(cv::norm(raw[c],before[c],cv::NORM_INF))return 8;}
   }
  }
  // Worker exceptions must be propagated after all readers have finished,
  // and the same processor must be reusable after a failed batch.
  auto good=Jobs(raw,out,4.0);cv::Mat invalid(48,64,CV_16U);auto bad=good;bad[4].raw=&invalid;
  bool caught=false;try{six.Apply(bad.data());}catch(const cv::Exception&){caught=true;}
  if(!caught)return 3;six.Apply(good.data());
  bad=good;bad[0].raw=&invalid;caught=false;try{six.Apply(bad.data());}catch(const cv::Exception&){caught=true;}
  if(!caught)return 4;six.Apply(good.data());
  // Independent processors may share the helper lanes concurrently.
  std::atomic<bool> concurrent_ok{true};
  auto concurrent=[&](int seed) {
   try {
    SixCameraIspProcessor local(format);std::array<WhiteBalance,6> reference_wb{};
    cv::Mat inputs[6],colors[6],expected[6];cv::RNG random(seed);
    for(int i=0;i<20;++i) {
     for(auto& r:inputs) {
      // Fresh WB instances must also handle a tiny frame with no samples.
      r.create(i==0?48:128,i==0?64:192,CV_8U);
      if(i==0)r.setTo(0);else random.fill(r,cv::RNG::UNIFORM,0,256);
     }
     auto j=Jobs(inputs,colors,4.0),ref_jobs=Jobs(inputs,expected,4.0);
     for(int c=0;c<6;++c) {
      const auto& job=ref_jobs[c];
      ApplyFastBalancedISPImpl(*job.raw,*job.output,reference_wb[c],job.name,
          job.sensor_gain,job.bayer,format,
          job.raw_bayer_nr_done,job.metadata_region);
     }
     local.Apply(j.data());
     for(int c=0;c<6;++c)
      if(cv::norm(colors[c],expected[c],cv::NORM_INF))concurrent_ok=false;
    }
   } catch(...) { concurrent_ok=false; }
  };
  std::thread one(concurrent,1),two(concurrent,2);one.join();two.join();
  if(!concurrent_ok)return 9;
  std::cout<<"VERIFY format="<<(uyvy?"UYVY":"BGR")<<" images="<<frames<<" maxdiff="<<maxdiff<<" worker_and_caller_exception_recovery=PASS split_scalar_reference=PASS raw_immutable=PASS concurrent_processors=PASS digital_gain=PASS external_output_buffer=PASS\n";
  return 0;
 }
 std::string dir=argc>2?argv[2]:".";
 cv::Mat planes[4];for(int i=0;i<4;++i){planes[i]=cv::imread(dir+"/raw_plane"+std::to_string(i)+".png",cv::IMREAD_GRAYSCALE);if(planes[i].empty())return 5;}
 raw[0]=planes[0];raw[1]=planes[1];SplitAndVBinHalfResLanes(planes[2],planes[3],raw+2);
 auto jobs=Jobs(raw,out,4.92307692308);std::vector<double> times;
 bool legacy=argc>1 && std::string(argv[1])=="legacy";
 for(int i=0;i<100;++i){auto t=Clock::now();if(legacy)Legacy(a,b,jobs);else six.Apply(jobs.data());double dt=ms(t);if(i>=20)times.push_back(dt);}
 std::sort(times.begin(),times.end());double sum=0;for(double v:times)sum+=v;
 std::cout<<"BENCH n="<<times.size()<<" avg_ms="<<sum/times.size()<<" median_ms="<<times[times.size()/2]<<" p95_ms="<<times[75]<<std::endl;
}
