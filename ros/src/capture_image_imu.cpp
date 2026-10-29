#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <glob.h>
#include <unistd.h>
#include <dirent.h>
#include <stdlib.h>
#include <string>
#include <stdio.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Temperature.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Header.h>
#include <boost/make_shared.hpp>
#include <array>
#include <map>
#include <mutex>
#include <opencv2/opencv.hpp>
#include "../../src/usb/uvc/camera_output.h"
#include "../../src/usb/uvc/six_camera_isp.h"
using namespace std;

const double g = 9.7887;

CYPERSTEREO_USE_NAMESPACE

// BGR8 messages the ISP renders into directly.  A message is reused only once
// roscpp and every intra-process subscriber have released it (use_count()==1,
// i.e. only the pool holds it), so the ~3.9 MB pixel buffer is neither copied
// by cv_bridge nor reallocated and zero-filled every frame.
class Bgr8MessagePool {
 public:
  sensor_msgs::ImagePtr Acquire(int height, int width) {
    for (const auto &msg : pool_)
      if (msg.use_count() == 1 && static_cast<int>(msg->height) == height &&
          static_cast<int>(msg->width) == width)
        return msg;
    sensor_msgs::ImagePtr msg = boost::make_shared<sensor_msgs::Image>();
    msg->height = height;
    msg->width = width;
    msg->encoding = sensor_msgs::image_encodings::BGR8;
    msg->is_bigendian = 0;
    msg->step = static_cast<uint32_t>(width) * 3;
    msg->data.resize(static_cast<size_t>(msg->step) * height);
    if (pool_.size() < kMaxPooled) pool_.push_back(msg);
    return msg;
  }

 private:
  static constexpr size_t kMaxPooled = 4;
  std::vector<sensor_msgs::ImagePtr> pool_;
};

static cv::Mat WrapBgr8(const sensor_msgs::ImagePtr &msg) {
  return cv::Mat(static_cast<int>(msg->height), static_cast<int>(msg->width),
                 CV_8UC3, msg->data.data(), msg->step);
}

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "capture_image_imu");
  ros::NodeHandle nh;
  // The four camera pipelines already run concurrently in the SDK's
  // persistent worker pool.  Letting OpenCV create another pool in every
  // camera job oversubscribes the CPU and increases frame latency.
  cv::setNumThreads(1);
  image_transport::ImageTransport it(nh);
  std::array<image_transport::Publisher, 6> image_pub;
  std::array<ros::Publisher, 6> temperature_pub;
  for (int i = 0; i < 6; ++i) {
    image_pub[i] = it.advertise("/cam" + std::to_string(i) + "/image_raw", 1000);
    temperature_pub[i] = nh.advertise<sensor_msgs::Temperature>(
        "/cam" + std::to_string(i) + "/temperature", 10);
  }
  ros::Publisher IMU_pub = nh.advertise<sensor_msgs::Imu>("imu0", 1000);
  int imu_seq = 1;
  std::shared_ptr<cyperstereo::uvc::device> cyperstereo_device{nullptr};
  if (!cyperstereo::FindCyperstereoDevices(cyperstereo_device)) {
    return 0;
  }
  cyperstereo::FrameInfo frame_info{};
  // Auto-select the camera profile (MT9V034, SC136HGS or AR0234) from the USB serial
  // prefix, or from UVC frame size when no SN is burned.
  const std::string serial_num =
      cyperstereo::uvc::get_serial_number(*cyperstereo_device);
  const cyperstereo::CameraProfile &profile =
      cyperstereo::SelectProfile(serial_num, *cyperstereo_device);
  frame_info.Init(profile);
  frame_info.framestream.serial_num = serial_num;
  const int num_cameras = profile.num_cameras;
  const int process_cameras =
      cyperstereo::SmartSensProcessedCameraCount(serial_num, num_cameras);
  const bool is_bayer = cyperstereo::IsSmartSensProfile(profile);
  ROS_INFO("camera: %s  serial: %s  %dx%d@%d  cameras: %d%s",
           profile.name,
           serial_num.empty() ? "(none)" : serial_num.c_str(),
           profile.frame_width, profile.frame_height, profile.fps, num_cameras,
           (process_cameras == 1 && num_cameras > 1)
               ? "  process: C1 only (S1 color SKU)"
               : "");
  ROS_INFO("ISP mode: %s", FastBalancedIspEnabled() ? "fast-balanced" : "quality-reference");
  cyperstereo::uvc::set_device_mode(
      *cyperstereo_device, profile.frame_width, profile.frame_height,
      static_cast<int>(cyperstereo::Format::YUYV), profile.fps,
      [&frame_info](const void *data, std::function<void()> continuation) {
        cyperstereo::SetStreamData(frame_info, data, continuation);
      });
  cyperstereo::uvc::start_streaming(*cyperstereo_device, 0);

  // Persistent per-camera buffers, swapped with the framestream planes under
  // the lock (O(1), like capture_image_imu) so the USB poll thread is never
  // blocked while we copy pixels. The *_color buffers hold the demosaiced BGR
  // output for the Bayer camera profiles and stay unused for MT9V034.
  cv::Mat left_image(profile.frame_height, profile.cam_width, CV_8U);
  cv::Mat right_image(profile.frame_height, profile.cam_width, CV_8U);
  cv::Mat left_front_image(profile.frame_height, profile.cam_width, CV_8U);
  cv::Mat right_front_image(profile.frame_height, profile.cam_width, CV_8U);
  Bgr8MessagePool color_pool[6];
  IspProcessor isp(CameraBgrIspMode()), half_isp(isp.mode());
  std::unique_ptr<SixCameraIspProcessor> six_isp;
  const bool use_six_scheduler = isp.UsesFastBalanced() &&
      !FastEnvTruthy("CYPERSTEREO_DISABLE_SIX_CAMERA_ISP");
  cv::Mat half_raw[4];
  bool six_announced = false;

  while (ros::ok()) {
    if (!WaitForCameraFrame(frame_info, [] { return ros::ok(); })) break;

    double image_timestamp = 0.0;
    RawTelemetrySnapshot raw_telemetry;
    CameraSensorSnapshot sensors;
    uint32_t hardware_version = 0;
    uint32_t software_version = 0;
    std::array<double, 4> camera_gain{{1.0, 1.0, 1.0, 1.0}};
    std::array<bool, 4> fpga_bayer_nr{};
    std::array<double, 2> c56_gain{{1.0, 1.0}}, c56_temperature{};
    bool c56_valid = false;
    cyperstereo::IMUStreamData imu_data{};

    {
      std::lock_guard<std::mutex> lock(frame_info.mtx);
      image_timestamp = frame_info.framestream.image_timestamp;
      raw_telemetry = frame_info.framestream.raw_telemetry;
      sensors = CameraSensorSnapshot::FromFrame(frame_info.framestream);
      hardware_version = frame_info.framestream.hardware_version;
      software_version = frame_info.framestream.software_version;
      cv::swap(frame_info.framestream.left_image, left_image);
      if (process_cameras >= 2)
        cv::swap(frame_info.framestream.right_image, right_image);
      if (process_cameras >= 4) {
        cv::swap(frame_info.framestream.left_front_image, left_front_image);
        cv::swap(frame_info.framestream.right_front_image, right_front_image);
      }
      if (is_bayer) {
        for (int i = 0; i < process_cameras; ++i) {
          camera_gain[i] = frame_info.framestream.isp_camera_gain[i];
          fpga_bayer_nr[i] = frame_info.framestream.fpga_bayer_nr[i];
        }
      }
      c56_valid = frame_info.framestream.c56_telemetry_valid;
      for (int i = 0; i < 2; ++i) {
        c56_gain[i] = frame_info.framestream.c56_camera_gain[i];
        c56_temperature[i] = frame_info.framestream.c56_camera_temperature[i];
      }
      imu_data = frame_info.framestream.imu;
    }

    const bool half_res_lanes = is_bayer && process_cameras >= 4 &&
        IsAr0234FamilyHardware(static_cast<int>(hardware_version)) &&
        IsSmartSensHalfResLaneLayout(static_cast<int>(hardware_version),
                                    static_cast<int>(software_version));
    const int output_cameras = half_res_lanes ? 6 : process_cameras;
    sensor_msgs::ImagePtr messages[6];
    if (is_bayer) {
      cv::Mat color[6];
      for (int i = 0; i < output_cameras; ++i) {
        const bool half = half_res_lanes && i >= 2;
        messages[i] = color_pool[i].Acquire(
            half ? SmartSensHalfResHeight(profile.frame_height) : profile.frame_height,
            half ? SmartSensHalfResWidth(profile.cam_width) : profile.cam_width);
        color[i] = WrapBgr8(messages[i]);
      }
      if (half_res_lanes) {
        if (!six_announced) {
          ROS_INFO("AR0234 six-camera output: /cam0..5/image_raw = C1..6; C3..6 are half-width/half-height");
          six_announced = true;
        }
        SplitAndVBinHalfResLanes(left_front_image, right_front_image, half_raw);
        const double gains[4] = {camera_gain[3], camera_gain[2], c56_gain[0], c56_gain[1]};
        const IspFrameJob jobs[] = {
            {left_image, color[0], "cam1", camera_gain[0], SelectDeliveredPlaneBayer(hardware_version, software_version, 0), fpga_bayer_nr[0], raw_telemetry.RegionForPlane(0)},
            {right_image, color[1], "cam2", camera_gain[1], SelectDeliveredPlaneBayer(hardware_version, software_version, 1), fpga_bayer_nr[1], raw_telemetry.RegionForPlane(1)},
            {half_raw[0], color[2], "half-cam3", gains[0], SelectCameraBayerConversion(hardware_version, software_version, SixCamHalfImageCamera(0))},
            {half_raw[1], color[3], "half-cam4", gains[1], SelectCameraBayerConversion(hardware_version, software_version, SixCamHalfImageCamera(1))},
            {half_raw[2], color[4], "half-cam5", gains[2], SelectCameraBayerConversion(hardware_version, software_version, SixCamHalfImageCamera(2))},
            {half_raw[3], color[5], "half-cam6", gains[3], SelectCameraBayerConversion(hardware_version, software_version, SixCamHalfImageCamera(3))}};
        if (use_six_scheduler) {
          if (!six_isp) six_isp.reset(new SixCameraIspProcessor());
          six_isp->Apply(jobs);
        } else {
          isp.ApplyParallel(jobs, 2);
          half_isp.ApplyParallel(jobs + 2, 4);
        }
        MirrorImage5Vertical(half_raw[2], color[4]);
      } else {
        cv::Mat *raws[4] = {&left_image, &right_image, &left_front_image, &right_front_image};
        cv::Mat *colors[4] = {&color[0], &color[1], &color[2], &color[3]};
        ApplyBayerCameraIsp(isp, raws, colors, process_cameras,
            hardware_version, software_version, camera_gain, fpga_bayer_nr, raw_telemetry);
      }
      for (int i = 0; i < output_cameras; ++i) {
        // Preserve the message-owned destination; copy only if an ISP backend
        // unexpectedly reallocates it. Subscribers may still own older frames.
        if (color[i].data != messages[i]->data.data())
          messages[i] = cv_bridge::CvImage(std_msgs::Header(), "bgr8", color[i]).toImageMsg();
      }
    } else {
      messages[0] = cv_bridge::CvImage(std_msgs::Header(), "mono8", left_image).toImageMsg();
      messages[1] = cv_bridge::CvImage(std_msgs::Header(), "mono8", right_image).toImageMsg();
    }
    for (int i = 0; i < output_cameras; ++i) {
      if (!messages[i]) continue;
      messages[i]->header.stamp = ros::Time(image_timestamp);
      messages[i]->header.frame_id = "cam" + std::to_string(i);
    }

    ROS_DEBUG("image_timestamp %.6f", image_timestamp);
    for (int i = 0; i < imu_data.imu_count; ++i) {
        double imu_timestamp = imu_data.imu_timestamp[i];
        double gyro_x = imu_data.gyro_x[i];
        double gyro_y = imu_data.gyro_y[i];
        double gyro_z = imu_data.gyro_z[i];
        double acc_x = imu_data.acc_x[i] * g;
        double acc_y = imu_data.acc_y[i] * g;
        double acc_z = imu_data.acc_z[i] * g;
        std::cout.setf(std::ios::fixed, std::ios::floatfield);
        std::cout.precision(6);
        ROS_DEBUG("imu_timestamp %.6f", imu_timestamp);

        sensor_msgs::Imu imu_msg;
        imu_msg.header.stamp = ros::Time(imu_timestamp);
        imu_msg.header.frame_id = "body";
        imu_msg.header.seq = imu_seq++;
        //acc  
        imu_msg.linear_acceleration.x = acc_x; 
        imu_msg.linear_acceleration.y = acc_y;
        imu_msg.linear_acceleration.z = acc_z;
        
        //gyro
        imu_msg.angular_velocity.x = gyro_x; 
        imu_msg.angular_velocity.y = gyro_y; 
        imu_msg.angular_velocity.z = gyro_z;

        IMU_pub.publish(imu_msg);
    }
    for (int i = 0; i < output_cameras; ++i) {
      const int sensor_index = half_res_lanes && i >= 2 && i < 4 ? 5 - i : i;
      const bool valid = i < 4 ? sensors.temperature_valid[sensor_index]
                              : c56_valid && std::isfinite(c56_temperature[i - 4]);
      if (valid) {
        sensor_msgs::Temperature temperature;
        temperature.header.stamp = ros::Time(image_timestamp);
        temperature.header.frame_id = "cam" + std::to_string(i);
        temperature.temperature = i < 4 ? sensors.temperature[sensor_index]
                                       : c56_temperature[i - 4];
        temperature.variance = 0.0;
        temperature_pub[i].publish(temperature);
      }
      if (messages[i]) image_pub[i].publish(messages[i]);
    }
    ros::spinOnce();
  }
  cyperstereo::uvc::stop_streaming(*cyperstereo_device);
  
  return 0;
}
