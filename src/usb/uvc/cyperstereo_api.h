#ifndef CYPERSTEREO_API_H_
#define CYPERSTEREO_API_H_

#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <sstream>
#include <mutex>
#include "string"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "uvc.h"
#include "tic_toc.h"

CYPERSTEREO_BEGIN_NAMESPACE

#define BMI088_ACCEL_24G_SEN 0.000732421875
#define BMI088_ACCEL_12G_SEN 0.0003662109375
#define BMI088_ACCEL_6G_SEN 0.00018310546875
#define BMI088_ACCEL_3G_SEN 0.000091552734375
#define BMI088_GYRO_2000_SEN 0.0010652644178602
#define BMI088_GYRO_1000_SEN 0.0005326322089301215
#define BMI088_GYRO_500_SEN 0.0002663161044650608
#define BMI088_GYRO_250_SEN 0.0001331580522325304
#define BMI088_GYRO_125_SEN 0.00006657902611626519
double BMI088_ACCEL_SEN = BMI088_ACCEL_6G_SEN;
double BMI088_GYRO_SEN = BMI088_GYRO_2000_SEN;


struct IMUStreamData {
  double acc_x[4];
	double acc_y[4];
	double acc_z[4];
	double gyro_x[4];
	double gyro_y[4];
	double gyro_z[4];
  double temperature[4];
  double imu_timestamp[4];
  int imu_count;
};

struct GNSSStreamData {
  bool valid;
  double gnss_timestamp;
  std::string gnss_utc_time;
  double latitude;
  double longitude;
  double altitude;
  // 0 - 无效 1 - 单点定位 2 - 差分定位 4 - RTK 固定解 5 - RTK 浮点解 6 - 惯导定位 50 - UWB 定位
  int fix_type;
  int satellites_used;
  double gps_geoid_height;
  double velocity;
  double heading;
  double hdop;
  double vdop;
  double pdop;
};

struct FrameStreamData {
  cv::Mat left_image;
  cv::Mat right_image;
  IMUStreamData imu;
  GNSSStreamData gnss;
	double image_timestamp;
};

struct Frame {
  const void *data = nullptr;
  std::function<void()> continuation = nullptr;
  Frame() {
  }
  ~Frame() {
    data = nullptr;
    if (continuation) {
      continuation();
      continuation = nullptr;
    }
  }
};

struct FrameInfo {
  std::mutex mtx;
  std::condition_variable con;
  std::shared_ptr<Frame> frame{nullptr};
  FrameStreamData framestream{};
  double last_imu_timestamp{0};
  double last_image_timestamp{0};
  double last_imu_count_s{0};
  double last_imu_count_ms{0};
  double last_image_count_s{0};
  FrameInfo() {
    framestream.left_image.create(480, 752, CV_8U);
    framestream.right_image.create(480, 752, CV_8U);
  }
};

cv::Mat FastGuidedfilter(cv::Mat &I, int r, float eps, int size) {
    r = r / size;
    int wsize = 2 * r + 1;
    I.convertTo(I, CV_32FC1, 1/255.0);

    cv::Mat small_I, small_p;
    cv::resize(I, small_I, I.size()/size, 0, 0, cv::INTER_AREA);
    small_p = small_I;

    cv::Mat mean_I, mean_p;
    cv::boxFilter(small_I, mean_I, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);
    mean_p = mean_I;

    cv::Mat mean_II, mean_Ip;
    mean_II = small_I.mul(small_I);
    cv::boxFilter(mean_II, mean_II, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);
    mean_Ip = mean_II;

    cv::Mat var_I, cov_Ip, mean_mul_I;
    mean_mul_I=mean_I.mul(mean_I);
    cv::subtract(mean_II, mean_mul_I, var_I);
    cov_Ip = var_I;
    
    cv::Mat a, b;
    cv::divide(cov_Ip, (var_I+eps),a);
    cv::subtract(mean_p, a.mul(mean_I), b);

    cv::Mat mean_a, mean_b;
    cv::boxFilter(a, mean_a, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);
    cv::boxFilter(b, mean_b, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);

    cv::resize(mean_a, mean_a, I.size());
    cv::resize(mean_b, mean_b, I.size());

    cv::Mat out = mean_a.mul(I) + mean_b;
    out.convertTo(out, CV_8UC1, 255);
    return out;
}


bool FindCyperstereoDevices(std::shared_ptr<uvc::device>& cyperstereo_device) {
  std::vector<std::shared_ptr<uvc::device>> cyperstereo_devices;

  auto context = uvc::create_context();
  auto devices = uvc::query_devices(context);
  if (devices.size() <= 0) {
    std::cout << "No devices :(" << std::endl;
    return false;
  }

  for (auto &&device : devices) {
    auto vid = uvc::get_vendor_id(*device);
    if (vid == Cyperstereo_VID) {
      cyperstereo_devices.push_back(device);
    }
  }

  // std::string dashes(80, '-');

  size_t n = cyperstereo_devices.size();
  if (n <= 0) {
    std::cout << "No Cyperstereo devices :(" << std::endl;
    return false;
  }

  std::cout  << "usb devices: " << std::endl;
  for (size_t i = 0; i < n; i++) {
    auto device = cyperstereo_devices[i];
    auto name = uvc::get_video_name(*device);
    auto vid = uvc::get_vendor_id(*device);
    auto pid = uvc::get_product_id(*device);
    std::cout << "  index: " << i << ", name: " << name << ", vid: 0x"
              << std::hex << vid << ", pid: 0x" << std::hex << pid << std::endl;
  }

  if (n <= 1) {
    cyperstereo_device = cyperstereo_devices[0];
    std::cout << "Only one Cyperstereo device, select index: 0" << std::endl;
  } else {
    while (true) {
      size_t i;
      std::cout << "There are " << n << " Cyperstereo devices, select index: " << std::endl;
      std::cin >> i;
      if (i >= n) {
        std::cout << "Index out of range :(" << std::endl;
        continue;
      }
      cyperstereo_device = cyperstereo_devices[i];
      break;
    }
  }
  return true;
}

void WaitForStream(FrameInfo& frame_info) {
  std::unique_lock<std::mutex> lock(frame_info.mtx);
  const auto frame_ready = [&frame_info]() { return frame_info.frame != nullptr; };
  if (frame_info.frame == nullptr) {
    if (!frame_info.con.wait_for(lock, std::chrono::seconds(5), frame_ready))
      throw std::runtime_error("Timeout waiting for frame.");
  }  
  frame_info.frame = nullptr;
}

void SetStreamData(FrameInfo& frame_info, const void *data, std::function<void()> continuation) {
  std::unique_lock<std::mutex> lock(frame_info.mtx);
        if (frame_info.frame == nullptr) {
          frame_info.frame = std::make_shared<struct Frame>();
        } else {
          if (frame_info.frame->continuation) {
            frame_info.frame->continuation();
          }
        }
        frame_info.frame->data = data;  // not copy here
        frame_info.frame->continuation = continuation;
        
        cv::Mat img(480, 752, CV_8UC2, const_cast<void *>(frame_info.frame->data));
        cv::Mat imu(1, 752 * 2, CV_8U);
        unsigned char* frame_p = img.ptr<unsigned char>(0, 0);
        unsigned char* left_p = frame_info.framestream.left_image.ptr<unsigned char>(0, 0);
	      unsigned char* right_p = frame_info.framestream.right_image.ptr<unsigned char>(0, 0);
        unsigned char* imu_p = imu.ptr<unsigned char>(0, 0);
        for (int i = 0; i < 752 * 480; i++) {
          left_p[i] = frame_p[2 * i + 1];
          right_p[i] = frame_p[2 * i];
          if (i >= 752 * 479 && i <= 752 * 479 + 42) {
            left_p[i] = frame_p[2 * (i - 752) + 1];
            right_p[i] = frame_p[2 * (i - 752)];
          }
        }
		    for (int i = 752 * 479 * 2, j = 0; i < 752 * 480 * 2; ++i, ++j) {
			    imu_p[j] = frame_p[i];
	      }
        // ISP(frame_info.framestream.left_image, frame_info.framestream.right_image);
        // image data
        uint32_t version1 = ((uint16_t)((imu.at<uchar>(0, 1)) << 8) | imu.at<uchar>(0, 0));
        uint32_t version2 = ((uint16_t)((imu.at<uchar>(0, 3)) << 8) | imu.at<uchar>(0, 2));
        bool image_count_hour_valid{false};
        if(version1 == 0 && version2 == 2) {
          image_count_hour_valid = true;
        }

        double image_count_hour = ((uint16_t)((imu.at<uchar>(0, 5)) << 8) | imu.at<uchar>(0, 4));
        double image_count_ms = ((uint16_t)((imu.at<uchar>(0, 7)) << 8) | imu.at<uchar>(0, 6));
        double image_count_s = ((uint16_t)((imu.at<uchar>(0, 9)) << 8) | imu.at<uchar>(0, 8));
        if(!image_count_hour_valid) {
          image_count_hour = 0;
        }

        frame_info.framestream.image_timestamp = image_count_hour * 12 * 3600 + image_count_s + image_count_ms / 10000.0;
        if (frame_info.framestream.image_timestamp - frame_info.last_image_timestamp < 0.015) {
          std::cout << "image time too small " << frame_info.last_image_timestamp << "  " << frame_info.framestream.image_timestamp <<std::endl;
        }
        if (frame_info.framestream.image_timestamp - frame_info.last_image_timestamp > 0.025) {
          std::cout << "image time too large " << frame_info.last_image_timestamp << "  " << frame_info.framestream.image_timestamp <<std::endl;
        }
        frame_info.last_image_timestamp = frame_info.framestream.image_timestamp;
        frame_info.last_image_count_s = image_count_s;
	
        //imu data
        for (int i = 0; i < 4; ++i) {
          double imu_count_ms = ((uint16_t)((imu.at<uchar>(0, 11 + i * 18)) << 8) | imu.at<uchar>(0, 10 + i * 18));
          double imu_count_s = ((uint16_t)((imu.at<uchar>(0, 13 + i * 18)) << 8) | imu.at<uchar>(0, 12 + i * 18));
          //std::cout << "i " << i << " imu_count_s " << imu_count_s << "   last_imu_count_s " << frame_info.last_imu_count_s << std::endl;
          
          frame_info.framestream.imu.imu_timestamp[i] = image_count_hour * 12 * 3600 + imu_count_s + imu_count_ms / 10000.0;
          frame_info.framestream.imu.acc_x[i] = ((int16_t)((imu.at<uchar>(0, 15 + i * 18)) << 8) | imu.at<uchar>(0, 14 + i * 18))* BMI088_ACCEL_SEN;
		      frame_info.framestream.imu.acc_y[i] = ((int16_t)((imu.at<uchar>(0, 17 + i * 18)) << 8) | imu.at<uchar>(0, 16 + i * 18))* BMI088_ACCEL_SEN;
		      frame_info.framestream.imu.acc_z[i] = ((int16_t)((imu.at<uchar>(0, 19 + i * 18)) << 8) | imu.at<uchar>(0, 18 + i * 18))* BMI088_ACCEL_SEN;
          frame_info.framestream.imu.gyro_x[i] = ((int16_t)((imu.at<uchar>(0, 21 + i * 18)) << 8) | imu.at<uchar>(0, 20 + i * 18))* BMI088_GYRO_SEN;
		      frame_info.framestream.imu.gyro_y[i] = ((int16_t)((imu.at<uchar>(0, 23 + i * 18)) << 8) | imu.at<uchar>(0, 22 + i * 18))* BMI088_GYRO_SEN;
		      frame_info.framestream.imu.gyro_z[i] = ((int16_t)((imu.at<uchar>(0, 25 + i * 18)) << 8) | imu.at<uchar>(0, 24 + i * 18))* BMI088_GYRO_SEN;
          frame_info.framestream.imu.temperature[i] = ((int16_t)((int16_t)((imu.at<uchar>(0, 27 + i * 18)) << 8) | imu.at<uchar>(0, 26 + i * 18)));
          if (frame_info.framestream.imu.temperature[i] > 1023) {
            frame_info.framestream.imu.temperature[i] = frame_info.framestream.imu.temperature[i] - 2048;
          } else {
            frame_info.framestream.imu.temperature[i] = frame_info.framestream.imu.temperature[i];
          }
          frame_info.framestream.imu.temperature[i] = frame_info.framestream.imu.temperature[i] * 0.125 + 23;
          //std::cout << "frame_info.framestream.imu.temperature[i] " << frame_info.framestream.imu.temperature[i] << std::endl;
          if (i == 3 && ((frame_info.framestream.imu.imu_timestamp[i] - frame_info.last_imu_timestamp > 0.006) ||  (frame_info.framestream.imu.imu_timestamp[i] - frame_info.last_imu_timestamp < 0.004) || (abs(frame_info.framestream.imu.temperature[i] - frame_info.framestream.imu.temperature[i - 1]) > 0.5))) {
	          continue;
          }
          frame_info.framestream.imu.imu_count = i;

          if (frame_info.framestream.imu.imu_timestamp[i] - frame_info.last_imu_timestamp > 0.0075) {
            std::cout << "imu time too large " << frame_info.last_imu_timestamp << "  " << frame_info.framestream.imu.imu_timestamp[i] << std::endl;
          }
          if (frame_info.framestream.imu.imu_timestamp[i] - frame_info.last_imu_timestamp <= 0) {
            std::cout << "imu time too small " << frame_info.last_imu_timestamp << "  " << frame_info.framestream.imu.imu_timestamp[i] << std::endl;
          }

          frame_info.last_imu_timestamp = frame_info.framestream.imu.imu_timestamp[i];
          frame_info.last_imu_count_ms = imu_count_ms;
          frame_info.last_imu_count_s = imu_count_s;
        }

        //GNGGA
        double gnss_count_ms = ((uint16_t)((imu.at<uchar>(0, 97)) << 8) | imu.at<uchar>(0, 96));
        double gnss_count_s = ((uint16_t)((imu.at<uchar>(0, 99)) << 8) | imu.at<uchar>(0, 98));
        double gnss_timestamp = image_count_hour * 12 * 3600 + gnss_count_s + gnss_count_ms / 10000.0;

        uint16_t gnss_utc_hour = ((uint16_t)((imu.at<uchar>(0, 101) >> 3)));
        uint16_t gps_utc_minute = ((uint16_t)((imu.at<uchar>(0, 100) >> 2)));
        uint16_t gps_utc_second = ((uint16_t)((imu.at<uchar>(0, 103) >> 2)));
        uint16_t gps_utc_second_ms = ((uint16_t)((imu.at<uchar>(0, 102) >> 1))) * 10;
        std::ostringstream gnss_utc_stream;
        gnss_utc_stream << std::setfill('0') << std::setw(2) << gnss_utc_hour << ":"
                        << std::setw(2) << gps_utc_minute << ":"
                        << std::setw(2) << gps_utc_second << "."
                        << std::setw(3) << gps_utc_second_ms;
        std::string gnss_utc_time = gnss_utc_stream.str();
        bool gnss_time_unchanged = (gnss_utc_time == frame_info.framestream.gnss.gnss_utc_time);
        frame_info.framestream.gnss.valid = false;

        double gps_lat_degree = ((uint16_t)((imu.at<uchar>(0, 105) >> 1))); 
        double gps_lat_minute = ((uint16_t)((imu.at<uchar>(0, 104) >> 2)));
        double gps_lat_fraction = ((uint32_t)(((imu.at<uchar>(0, 107)) << 9) | (imu.at<uchar>(0, 106) << 1) | ((imu.at<uchar>(0, 104) & 0x02) >> 1)));
        uint16_t gps_lat_is_north = ((uint16_t)(imu.at<uchar>(0, 104) & 0x01));
        double latitude = gps_lat_degree + (gps_lat_minute + gps_lat_fraction / 100000.0) / 60.0;
        if (gps_lat_is_north == 0) {
          latitude = -latitude;
        }
        double gps_lon_degree = ((uint16_t)((imu.at<uchar>(0, 109))));
        double gps_lon_minute = ((uint16_t)((imu.at<uchar>(0, 108) >> 2)));
        double gps_lon_fraction = ((uint32_t)((imu.at<uchar>(0, 111) << 9) | (imu.at<uchar>(0, 110) << 1) | ((imu.at<uchar>(0, 108) & 0x02) >> 1)));
        uint16_t gps_lon_is_east = ((uint16_t)(imu.at<uchar>(0, 108) & 0x01));
        double longitude = gps_lon_degree + (gps_lon_minute + gps_lon_fraction / 100000.0) / 60.0;
        if (gps_lon_is_east == 0) {
          longitude = -longitude;
        }
        double gps_alt_integer = ((uint16_t)((imu.at<uchar>(0, 113) << 6 | imu.at<uchar>(0, 112) >> 2)));
        double gps_alt_fraction = (uint16_t)((imu.at<uchar>(0, 115) << 8) | imu.at<uchar>(0, 114));
        double altitude = gps_alt_integer + gps_alt_fraction / 10000.0;
        uint16_t gps_fix_quality = (uint16_t)(imu.at<uchar>(0, 117) >> 5);
        uint16_t gps_satellites = (uint16_t)(imu.at<uchar>(0, 116) >> 1);
        double gps_geoid_integer = (uint16_t)((imu.at<uchar>(0, 119) << 6) | (imu.at<uchar>(0, 118)  >> 2) );
        double gps_geoid_fraction = (uint16_t)((imu.at<uchar>(0, 121) << 4) | (imu.at<uchar>(0, 120)  >> 4) );
        uint16_t gps_geoid_negative = (imu.at<uchar>(0, 118) & 0x01);
        double gps_geoid_height = gps_geoid_integer + gps_geoid_fraction / 10000.0;
        if (gps_geoid_negative == 1) {
          gps_geoid_height = -gps_geoid_height;
        }
        //GNGST
        double gps_lat_err_integer = (uint16_t)((imu.at<uchar>(0, 123) << 8) | imu.at<uchar>(0, 122));
        double gps_lat_err_fraction = (uint16_t)((imu.at<uchar>(0, 125) << 2) | imu.at<uchar>(0, 124) >> 6);
        double gps_lat_err = gps_lat_err_integer + gps_lat_err_fraction / 1000.0;
        double gps_lon_err_integer = (uint16_t)((imu.at<uchar>(0, 127) << 8) | imu.at<uchar>(0, 126));
        double gps_lon_err_fraction = (uint16_t)((imu.at<uchar>(0, 129) << 2) | imu.at<uchar>(0, 128) >> 6);
        double gps_lon_err = gps_lon_err_integer + gps_lon_err_fraction / 1000.0;
        double gps_alt_err_integer = (uint16_t)((imu.at<uchar>(0, 131) << 8) | imu.at<uchar>(0, 130));
        double gps_alt_err_fraction = (uint16_t)((imu.at<uchar>(0, 133) << 2) | imu.at<uchar>(0, 132) >> 6);
        double gps_alt_err = gps_alt_err_integer + gps_alt_err_fraction / 1000.0;
        double hdop = sqrt(gps_lat_err * gps_lat_err + gps_lon_err * gps_lon_err);
        double vdop = gps_alt_err;
        double pdop = sqrt(hdop * hdop + vdop * vdop);
        //GNRMC
        double gps_speed_kmh_integer = (uint16_t)((imu.at<uchar>(0, 135) << 8) | imu.at<uchar>(0, 134));
        double gps_speed_kmh_fraction = (uint16_t)((imu.at<uchar>(0, 137) << 2) | imu.at<uchar>(0, 136) >> 6);
        double gps_speed_kmh = gps_speed_kmh_integer + gps_speed_kmh_fraction / 1000.0;
        double gps_angle_integer = (uint16_t)((imu.at<uchar>(0, 139) << 8) | imu.at<uchar>(0, 138));
        double gps_angle_fraction = (uint16_t)((imu.at<uchar>(0, 141) << 2) | imu.at<uchar>(0, 140) >> 6);
        double gps_angle = gps_angle_integer + gps_angle_fraction / 1000.0;
        
        if (!gnss_time_unchanged && gps_fix_quality >=1 && hdop < 50 && vdop < 50 && pdop < 50) {
          frame_info.framestream.gnss.valid = true;
          frame_info.framestream.gnss.gnss_timestamp = gnss_timestamp;
          frame_info.framestream.gnss.gnss_utc_time = gnss_utc_time;
          frame_info.framestream.gnss.latitude = latitude;
          frame_info.framestream.gnss.longitude = longitude;
          frame_info.framestream.gnss.altitude = altitude;
          frame_info.framestream.gnss.fix_type = gps_fix_quality;
          frame_info.framestream.gnss.satellites_used = gps_satellites;
          frame_info.framestream.gnss.gps_geoid_height = gps_geoid_height;
          frame_info.framestream.gnss.hdop = hdop;
          frame_info.framestream.gnss.vdop = vdop;
          frame_info.framestream.gnss.pdop = pdop;
          frame_info.framestream.gnss.velocity = gps_speed_kmh;
          frame_info.framestream.gnss.heading = gps_angle;
        }

        //if (frame_info.frame != nullptr)
        frame_info.con.notify_one();
}

CYPERSTEREO_END_NAMESPACE
#endif  // CYPERSTEREO_API_H_
