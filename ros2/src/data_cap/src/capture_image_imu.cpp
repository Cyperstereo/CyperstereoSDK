#include <glob.h>
#include <unistd.h>
#include <dirent.h>
#include <stdlib.h>
#include <stdio.h>

#include <map>
#include <array>
#include <atomic>
#include <string>
#include <memory>
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <mutex>
#include <thread>

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
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr cam2_image_pub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr cam3_image_pub;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub;

    rclcpp::TimerBase::SharedPtr timer_;

    std::thread worker_thread_;
    std::atomic<bool> is_running_;

public:
    CameraImuPublisher() : Node("data_publisher")
    {
        // The SDK parallelises the four cameras with persistent workers.
        // Disable nested OpenCV worker pools to avoid CPU oversubscription.
        cv::setNumThreads(1);
        cam0_image_pub = this->create_publisher<sensor_msgs::msg::Image>("/cam0/image_raw", 10);
        cam1_image_pub = this->create_publisher<sensor_msgs::msg::Image>("/cam1/image_raw", 10);
        cam2_image_pub = this->create_publisher<sensor_msgs::msg::Image>("/cam2/image_raw", 10);
        cam3_image_pub = this->create_publisher<sensor_msgs::msg::Image>("/cam3/image_raw", 10);
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

        std::shared_ptr<cyperstereo::uvc::device> cyperstereo_device{nullptr};
        if (!cyperstereo::FindCyperstereoDevices(cyperstereo_device))
        {
            return;
        }
        cyperstereo::FrameInfo frame_info{};
        // Auto-select the camera profile (MT9V034 vs SmartSens) from the USB
        // serial prefix, or from UVC frame size when no SN is burned.
        const std::string serial_num =
            cyperstereo::uvc::get_serial_number(*cyperstereo_device);
        const cyperstereo::CameraProfile &profile =
            cyperstereo::SelectProfile(serial_num, *cyperstereo_device);
        frame_info.Init(profile);
        frame_info.framestream.serial_num = serial_num;
        const int num_cameras = profile.num_cameras;
        const int process_cameras =
            cyperstereo::SmartSensProcessedCameraCount(serial_num, num_cameras);
        const bool is_smartsens = cyperstereo::IsSmartSensProfile(profile);
        RCLCPP_INFO(this->get_logger(),
                    "camera: %s  serial: %s  %dx%d@%d  cameras: %d%s",
                    profile.name,
                    serial_num.empty() ? "(none)" : serial_num.c_str(),
                    profile.frame_width, profile.frame_height, profile.fps,
                    num_cameras,
                    (process_cameras == 1 && num_cameras > 1)
                        ? "  process: C1 only (S1 color SKU)"
                        : "");
        cyperstereo::uvc::set_device_mode(
            *cyperstereo_device, profile.frame_width, profile.frame_height,
            static_cast<int>(cyperstereo::Format::YUYV), profile.fps,
            [&frame_info](const void *data, std::function<void()> continuation)
            {
                cyperstereo::SetStreamData(frame_info, data, continuation);
            });

        cyperstereo::uvc::start_streaming(*cyperstereo_device, 0);

        // Persistent per-camera buffers, swapped with the framestream planes
        // under the lock (O(1), like capture_image_imu) so the USB poll thread
        // is never blocked while we copy pixels. The *_color buffers hold the
        // demosaiced BGR output for the 4-camera (SmartSens) profile and stay
        // unused for MT9V034.
        cv::Mat left_image(profile.frame_height, profile.cam_width, CV_8U);
        cv::Mat right_image(profile.frame_height, profile.cam_width, CV_8U);
        cv::Mat left_front_image(profile.frame_height, profile.cam_width, CV_8U);
        cv::Mat right_front_image(profile.frame_height, profile.cam_width, CV_8U);
        cv::Mat left_color, right_color, left_front_color, right_front_color;

        while (is_running_)
        {
            cyperstereo::WaitForStream(frame_info);

            double image_timestamp = 0.0;
            uint32_t hardware_version = 0;
            uint32_t software_version = 0;
            std::array<double, 4> camera_gain{{1.0, 1.0, 1.0, 1.0}};
            cyperstereo::IMUStreamData imu_data{};

            {
                std::lock_guard<std::mutex> lock(frame_info.mtx);
                image_timestamp = frame_info.framestream.image_timestamp;
                hardware_version = frame_info.framestream.hardware_version;
                software_version = frame_info.framestream.software_version;
                cv::swap(frame_info.framestream.left_image, left_image);
                if (process_cameras >= 2)
                    cv::swap(frame_info.framestream.right_image, right_image);
                if (process_cameras >= 4) {
                    cv::swap(frame_info.framestream.left_front_image, left_front_image);
                    cv::swap(frame_info.framestream.right_front_image, right_front_image);
                }
                if (is_smartsens) {
                    for (int i = 0; i < process_cameras; ++i)
                        camera_gain[i] = frame_info.framestream.camera_gain[i];
                }
                imu_data = frame_info.framestream.imu;
            }

            cv_bridge::CvImage msg0, msg1, msg2, msg3;
            rclcpp::Time custom_time(image_timestamp*1e9); // seconds -> ns

            if (is_smartsens)
            {
                // SmartSens: use the SDK's persistent fast-balanced worker
                // pool instead of constructing three threads every frame.
                // Keep independent AWB state for every sensor. Cameras 1 and
                // 3 use the hardware-dependent Bayer phase, while 2 and 4
                // retain the default RG2BGR mapping.
                static WhiteBalance wb[4];
                const BayerConversion image13_bayer =
                    SelectBayerConversion(hardware_version, software_version, 0);
                if (process_cameras <= 1) {
                    // S1: C1-only color SKU. Unused FX3 lanes are not ISP'd
                    // or published.
                    ApplyFastBalancedISPParallel({
                        {left_image, left_color, wb[0], "fast-cam1",
                         camera_gain[0], image13_bayer},
                    });
                } else if (process_cameras >= 4) {
                    ApplyFastBalancedISPParallel({
                        {left_image, left_color, wb[0], "fast-cam1",
                         camera_gain[0], image13_bayer},
                        {right_image, right_color, wb[1], "fast-cam2",
                         camera_gain[1]},
                        {left_front_image, left_front_color, wb[2], "fast-cam3",
                         camera_gain[2], image13_bayer},
                        {right_front_image, right_front_color, wb[3], "fast-cam4",
                         camera_gain[3]},
                    });
                } else {
                    ApplyFastBalancedISPParallel({
                        {left_image, left_color, wb[0], "fast-cam1",
                         camera_gain[0], image13_bayer},
                        {right_image, right_color, wb[1], "fast-cam2",
                         camera_gain[1]},
                    });
                }

                msg0.encoding = "bgr8";
                msg0.image = left_color;
                if (process_cameras >= 2) {
                    msg1.encoding = "bgr8";
                    msg1.image = right_color;
                }

                if (process_cameras >= 4) {
                    msg2.header.stamp = custom_time;
                    msg2.header.frame_id = "cam2";
                    msg2.encoding = "bgr8";
                    msg2.image = left_front_color;

                    msg3.header.stamp = custom_time;
                    msg3.header.frame_id = "cam3";
                    msg3.encoding = "bgr8";
                    msg3.image = right_front_color;
                }
            }
            else
            {
                // MT9V034 is monochrome: publish the two raw planes as mono8.
                msg0.encoding = "mono8";
                msg0.image = left_image;
                msg1.encoding = "mono8";
                msg1.image = right_image;
            }

            msg0.header.stamp = custom_time;
            msg0.header.frame_id = "cam0";
            if (process_cameras >= 2) {
                msg1.header.stamp = custom_time;
                msg1.header.frame_id = "cam1";
            }

            std::cout << "image_timestamp " << image_timestamp << std::endl;
            
            for (int i = 0; i < imu_data.imu_count; ++i)
            {
                double imu_timestamp = imu_data.imu_timestamp[i];
                double gyro_x = imu_data.gyro_x[i];
                double gyro_y = imu_data.gyro_y[i];
                double gyro_z = imu_data.gyro_z[i];
                double acc_x = imu_data.acc_x[i] * g;
                double acc_y = imu_data.acc_y[i] * g;
                double acc_z = imu_data.acc_z[i] * g;
                std::cout.setf(std::ios::fixed, std::ios::floatfield);
                std::cout.precision(6);
                std::cout << "imu_timestamp " << imu_timestamp << " " << gyro_x << " " << gyro_y << " " << gyro_z << " " << acc_x << " " << acc_y << " " << acc_z << std::endl;

                sensor_msgs::msg::Imu imu_msg;
                imu_msg.header.stamp = rclcpp::Time(imu_timestamp*1e9);
                imu_msg.header.frame_id = "imu0";

                // acc
                imu_msg.linear_acceleration.x = acc_x;
                imu_msg.linear_acceleration.y = acc_y;
                imu_msg.linear_acceleration.z = acc_z;

                // gyro
                imu_msg.angular_velocity.x = gyro_x;
                imu_msg.angular_velocity.y = gyro_y;
                imu_msg.angular_velocity.z = gyro_z;

                imu_pub->publish(imu_msg);
            }

            cam0_image_pub->publish(*msg0.toImageMsg());
            if (process_cameras >= 2)
                cam1_image_pub->publish(*msg1.toImageMsg());
            if (process_cameras >= 4)
            {
                cam2_image_pub->publish(*msg2.toImageMsg());
                cam3_image_pub->publish(*msg3.toImageMsg());
            }
            
        }
        cyperstereo::uvc::stop_streaming(*cyperstereo_device);
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CameraImuPublisher>());
    rclcpp::shutdown();
    return 0;
}
