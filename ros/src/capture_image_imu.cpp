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
#include <map>
#include <exception>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "../../src/usb/uvc/cyperstereo_api.h"
using namespace std;

const double g = 9.7887;

CYPERSTEREO_USE_NAMESPACE

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "capture_image_imu");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  image_transport::Publisher cam0_image_pub = it.advertise("/cam0/image_raw", 1000);
  image_transport::Publisher cam1_image_pub = it.advertise("/cam1/image_raw", 1000);
  ros::Publisher IMU_pub = nh.advertise<sensor_msgs::Imu>("imu0", 1000); 

  int count = 0;
  int imu_seq = 1;
  while (ros::ok()) {
    std::shared_ptr<cyperstereo::uvc::device> cyperstereo_device{nullptr};
    if (!cyperstereo::FindCyperstereoDevices(cyperstereo_device)) {
      ROS_WARN("No Cyperstereo device, retrying in 1s...");
      ros::Duration(1.0).sleep();
      continue;
    }

    cyperstereo::FrameInfo frame_info{};
    frame_info.ResetState();
    cyperstereo::uvc::set_device_mode(
        *cyperstereo_device, 752, 480, static_cast<int>(cyperstereo::Format::YUYV), 60,
        [&frame_info](const void *data, std::function<void()> continuation) {
          cyperstereo::SetStreamData(frame_info, data, continuation);
        });

    try {
      cyperstereo::uvc::start_streaming(*cyperstereo_device, 0);
      while (ros::ok()) {
        cyperstereo::WaitForStream(frame_info);
        double image_timestamp = frame_info.framestream.image_timestamp;
        cv::Mat left_image = frame_info.framestream.left_image;
        sensor_msgs::ImagePtr msg0 = cv_bridge::CvImage(std_msgs::Header(), "mono8", left_image).toImageMsg();
        msg0->header.stamp = ros::Time(image_timestamp);

        cv::Mat right_image = frame_info.framestream.right_image;
        sensor_msgs::ImagePtr msg1 = cv_bridge::CvImage(std_msgs::Header(), "mono8", right_image).toImageMsg();
        msg1->header.stamp = ros::Time(image_timestamp);

        if (count++ % 2 != 0) {
          std::cout << "image_timestamp " << image_timestamp << std::endl;
          // cv::imshow("left", left_image);
          // cv::imshow("right", right_image);
          // cv::waitKey(1);
        } 
        for (int i = 0; i <= frame_info.framestream.imu.imu_count; ++i) {
          double imu_timestamp = frame_info.framestream.imu.imu_timestamp[i];
          double gyro_x = frame_info.framestream.imu.gyro_x[i];
          double gyro_y = frame_info.framestream.imu.gyro_y[i];
          double gyro_z = frame_info.framestream.imu.gyro_z[i];
          double acc_x = frame_info.framestream.imu.acc_x[i] * g;
          double acc_y = frame_info.framestream.imu.acc_y[i] * g;
          double acc_z = frame_info.framestream.imu.acc_z[i] * g;
          std::cout.setf(std::ios::fixed, std::ios::floatfield);
          std::cout.precision(6);
          std::cout << "imu_timestamp " << imu_timestamp << " " << gyro_x << " "<< gyro_y << " "<< gyro_z << " " << acc_x << " "<< acc_y << " " << acc_z << std::endl;

          sensor_msgs::Imu imu_data;
          imu_data.header.stamp = ros::Time(imu_timestamp);
          imu_data.header.frame_id = "body";
          imu_data.header.seq = imu_seq++;
          //acc  
          imu_data.linear_acceleration.x = acc_x; 
          imu_data.linear_acceleration.y = acc_y;
          imu_data.linear_acceleration.z = acc_z;
          
          //gyro
          imu_data.angular_velocity.x = gyro_x; 
          imu_data.angular_velocity.y = gyro_y; 
          imu_data.angular_velocity.z = gyro_z;

          IMU_pub.publish(imu_data);
        }
        cam0_image_pub.publish(msg0);
        cam1_image_pub.publish(msg1);
        ros::spinOnce();
      }
      cyperstereo::uvc::stop_streaming(*cyperstereo_device);
      break;
    } catch (const std::exception &e) {
      std::cerr << "capture loop error: " << e.what() << ", restarting when device is back." << std::endl;
      cyperstereo::uvc::stop_streaming(*cyperstereo_device);
      ros::Duration(1.0).sleep();
    }
  }

  return 0;
}
