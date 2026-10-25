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
  image_transport::Publisher cam0_image_pub = it.advertise("/cam0/image_raw", 1000);
  image_transport::Publisher cam1_image_pub = it.advertise("/cam1/image_raw", 1000);
  image_transport::Publisher cam2_image_pub = it.advertise("/cam2/image_raw", 1000);
  image_transport::Publisher cam3_image_pub = it.advertise("/cam3/image_raw", 1000);
  ros::Publisher IMU_pub = nh.advertise<sensor_msgs::Imu>("imu0", 1000); 

  std::array<ros::Publisher, 4> temperature_pub;
  for (int i = 0; i < 4; ++i)
    temperature_pub[i] = nh.advertise<sensor_msgs::Temperature>(
        "/cam" + std::to_string(i) + "/temperature", 10);
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
  Bgr8MessagePool color_pool[4];

  while (ros::ok()) {
    if (!WaitForCameraFrame(frame_info, [] { return ros::ok(); })) break;

    double image_timestamp = 0.0;
    RawTelemetrySnapshot raw_telemetry;
    CameraSensorSnapshot sensors;
    uint32_t hardware_version = 0;
    uint32_t software_version = 0;
    std::array<double, 4> camera_gain{{1.0, 1.0, 1.0, 1.0}};
    std::array<bool, 4> fpga_bayer_nr{};
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
      imu_data = frame_info.framestream.imu;
    }

    sensor_msgs::ImagePtr msg0, msg1, msg2, msg3;
    if (is_bayer) {
      // SC136HGS / AR0234: each plane is RAW Bayer with no on-chip AWB.  The SDK's
      // fast-balanced path owns three persistent workers and runs the fourth
      // camera on this thread, avoiding three thread creations every frame.
      // Each camera keeps independent AWB history.  Cameras 1 and 3 share the
      // hardware-version-dependent Bayer phase; all phases come from the profile.
      static IspProcessor isp(CameraBgrIspMode());
      sensor_msgs::ImagePtr *const color_msg[4] = {&msg0, &msg1, &msg2, &msg3};
      cv::Mat color[4];
      for (int i = 0; i < process_cameras && i < 4; ++i) {
        *color_msg[i] =
            color_pool[i].Acquire(profile.frame_height, profile.cam_width);
        color[i] = WrapBgr8(*color_msg[i]);
      }
      cv::Mat *raws[4] = {&left_image, &right_image, &left_front_image, &right_front_image};
      cv::Mat *colors[4] = {&color[0], &color[1], &color[2], &color[3]};
      ApplyBayerCameraIsp(isp, raws, colors, process_cameras,
          hardware_version, software_version, camera_gain, fpga_bayer_nr, raw_telemetry);
      for (int i = 0; i < process_cameras && i < 4; ++i) {
        // The ISP keeps the wrapped buffer when size/type match; if it ever
        // had to reallocate, publish a copy of what it produced instead.
        if (color[i].data != (*color_msg[i])->data.data())
          *color_msg[i] = cv_bridge::CvImage(std_msgs::Header(), "bgr8",
                                             color[i]).toImageMsg();
      }
      if (process_cameras >= 4) {
        msg2->header.stamp = ros::Time(image_timestamp);
        msg3->header.stamp = ros::Time(image_timestamp);
      }
    } else {
      // MT9V034 is monochrome: publish the two raw planes as mono8.
      msg0 = cv_bridge::CvImage(std_msgs::Header(), "mono8", left_image).toImageMsg();
      msg1 = cv_bridge::CvImage(std_msgs::Header(), "mono8", right_image).toImageMsg();
    }
    msg0->header.stamp = ros::Time(image_timestamp);
    if (msg1)
      msg1->header.stamp = ros::Time(image_timestamp);

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
    sensor_msgs::ImagePtr messages[4] = {msg0, msg1, msg2, msg3};
    for (int i = 0; i < process_cameras; ++i) {
      if (messages[i]) messages[i]->header.frame_id = "cam" + std::to_string(i);
      if (!sensors.temperature_valid[i]) continue;
      sensor_msgs::Temperature temperature;
      temperature.header.stamp = ros::Time(image_timestamp);
      temperature.header.frame_id = "cam" + std::to_string(i);
      temperature.temperature = sensors.temperature[i];
      temperature.variance = 0.0;  // unknown uncertainty
      temperature_pub[i].publish(temperature);
    }
    cam0_image_pub.publish(msg0);
    if (process_cameras >= 2)
      cam1_image_pub.publish(msg1);
    if (process_cameras >= 4) {
      cam2_image_pub.publish(msg2);
      cam3_image_pub.publish(msg3);
    }
    ros::spinOnce();
  }
  cyperstereo::uvc::stop_streaming(*cyperstereo_device);
  
  return 0;
}
