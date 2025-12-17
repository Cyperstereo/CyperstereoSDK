#include <glob.h>
#include <unistd.h>
#include <dirent.h>
#include <stdlib.h>
#include <stdio.h>

#include <map>
#include <string>
#include <memory>
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <exception>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/imu.hpp"

#include "../../../../src/usb/uvc/cyperstereo_api.h"

const double g = 9.7887;
CYPERSTEREO_USE_NAMESPACE

using namespace std;
using namespace std::chrono_literals;

class CameraImuPublisher : public rclcpp::Node
{

private:
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr cam0_image_pub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr cam1_image_pub;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub;

    rclcpp::TimerBase::SharedPtr timer_;

    std::thread worker_thread_;
    std::atomic<bool> is_running_;

public:
    CameraImuPublisher() : Node("data_publisher")
    {
        cam0_image_pub = this->create_publisher<sensor_msgs::msg::Image>("/cam0/image_raw", 10);
        cam1_image_pub = this->create_publisher<sensor_msgs::msg::Image>("/cam1/image_raw", 10);
        imu_pub = this->create_publisher<sensor_msgs::msg::Imu>("/imu0", 1000);
        is_running_ = true;
        worker_thread_ = std::thread(&CameraImuPublisher::publish_data, this);

        RCLCPP_INFO(this->get_logger(), "Data publisher node initialized.");
    }

    ~CameraImuPublisher()
    {
        is_running_ = false;

        if (worker_thread_.joinable())
        {
            worker_thread_.join();
            RCLCPP_INFO(this->get_logger(), "Data publisher node thread exit.");
        }
    }

    void publish_data()
    {
        while (is_running_)
        {
            std::shared_ptr<cyperstereo::uvc::device> cyperstereo_device{nullptr};
            if (!cyperstereo::FindCyperstereoDevices(cyperstereo_device))
            {
                RCLCPP_WARN(this->get_logger(), "No Cyperstereo device, retrying in 1s...");
                std::this_thread::sleep_for(1s);
                continue;
            }

            cyperstereo::FrameInfo frame_info{};
            frame_info.ResetState();
            cyperstereo::uvc::set_device_mode(
                *cyperstereo_device, 752, 480, static_cast<int>(cyperstereo::Format::YUYV), 60,
                [&frame_info](const void *data, std::function<void()> continuation)
                {
                    cyperstereo::SetStreamData(frame_info, data, continuation);
                });

            try
            {
                cyperstereo::uvc::start_streaming(*cyperstereo_device, 0);
                while (is_running_)
                {
                    cyperstereo::WaitForStream(frame_info);
                    double image_timestamp = frame_info.framestream.image_timestamp;
                    cv::Mat left_image = frame_info.framestream.left_image;
                    cv::Mat right_image = frame_info.framestream.right_image;

                    cv_bridge::CvImage msg0, msg1;

                    rclcpp::Time custom_time(image_timestamp*1e9); // seconds, ns or ns
                    msg0.header.stamp = custom_time;
                    msg0.header.frame_id = "cam0";
                    msg0.encoding = "mono8"; //"bgr8";
                    msg0.image = left_image;

                    msg1.header.stamp = custom_time;
                    msg1.header.frame_id = "cam1";
                    msg1.encoding = "mono8"; //"bgr8";
                    msg1.image = right_image;

                    std::cout << "image_timestamp " << image_timestamp << std::endl;
                    
                    for (int i = 0; i <= frame_info.framestream.imu.imu_count; ++i)
                    {
                        double imu_timestamp = frame_info.framestream.imu.imu_timestamp[i];
                        double gyro_x = frame_info.framestream.imu.gyro_x[i];
                        double gyro_y = frame_info.framestream.imu.gyro_y[i];
                        double gyro_z = frame_info.framestream.imu.gyro_z[i];
                        double acc_x = frame_info.framestream.imu.acc_x[i] * g;
                        double acc_y = frame_info.framestream.imu.acc_y[i] * g;
                        double acc_z = frame_info.framestream.imu.acc_z[i] * g;
                        std::cout.setf(std::ios::fixed, std::ios::floatfield);
                        std::cout.precision(6);
                        std::cout << "imu_timestamp " << imu_timestamp << " " << gyro_x << " " << gyro_y << " " << gyro_z << " " << acc_x << " " << acc_y << " " << acc_z << std::endl;

                        sensor_msgs::msg::Imu imu_data;
                        imu_data.header.stamp = rclcpp::Time(imu_timestamp*1e9);
                        imu_data.header.frame_id = "imu0";

                        // acc
                        imu_data.linear_acceleration.x = acc_x;
                        imu_data.linear_acceleration.y = acc_y;
                        imu_data.linear_acceleration.z = acc_z;

                        // gyro
                        imu_data.angular_velocity.x = gyro_x;
                        imu_data.angular_velocity.y = gyro_y;
                        imu_data.angular_velocity.z = gyro_z;

                        imu_pub->publish(imu_data);
                    }

                    cam0_image_pub->publish(*msg0.toImageMsg());
                    cam1_image_pub->publish(*msg1.toImageMsg());
                    
                }
                cyperstereo::uvc::stop_streaming(*cyperstereo_device);
                break;
            }
            catch (const std::exception &e)
            {
                RCLCPP_WARN(this->get_logger(), "capture loop error: %s, restarting when device is back.", e.what());
                cyperstereo::uvc::stop_streaming(*cyperstereo_device);
                std::this_thread::sleep_for(1s);
            }
        }
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CameraImuPublisher>());
    rclcpp::shutdown();
    return 0;
}
