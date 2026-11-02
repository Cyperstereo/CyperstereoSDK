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
#include "sensor_msgs/msg/temperature.hpp"

#include "../../../../src/usb/uvc/camera_output.h"
#include "../../../../src/usb/uvc/six_camera_isp.h"

const double g = 9.7887;
CYPERSTEREO_USE_NAMESPACE

using namespace std;
using namespace std::chrono_literals;

class CameraImuPublisher : public rclcpp::Node
{

private:
    std::array<rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr, 6> cam_image_pub;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub;

    std::array<rclcpp::Publisher<sensor_msgs::msg::Temperature>::SharedPtr, 6> temperature_pub;

    std::thread worker_thread_;
    std::atomic<bool> is_running_;

public:
    CameraImuPublisher() : Node("data_publisher")
    {
        // The SDK parallelises the four cameras with persistent workers.
        // Disable nested OpenCV worker pools to avoid CPU oversubscription.
        cv::setNumThreads(1);
        for (int i = 0; i < 6; ++i)
            cam_image_pub[i] = this->create_publisher<sensor_msgs::msg::Image>(
                "/cam" + std::to_string(i) + "/image_raw", 10);
        imu_pub = this->create_publisher<sensor_msgs::msg::Imu>("/imu0", 1000);
        for (int i = 0; i < 6; ++i)
            temperature_pub[i] = this->create_publisher<sensor_msgs::msg::Temperature>(
                "/cam" + std::to_string(i) + "/temperature", 10);
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
        // Auto-select the camera profile (MT9V034, SC136HGS or AR0234) from the USB
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
        const bool is_bayer = cyperstereo::IsSmartSensProfile(profile);
        // S6/A6 serial = six-image FPGA build (column 0 is the same as the quad).
        const bool six_image = IsSixImageSerial(serial_num);
        RCLCPP_INFO(this->get_logger(),
                    "camera: %s  serial: %s  %dx%d@%d  cameras: %d%s",
                    profile.name,
                    serial_num.empty() ? "(none)" : serial_num.c_str(),
                    profile.frame_width, profile.frame_height, profile.fps,
                    CameraModeImageCount(six_image, num_cameras),
                    (process_cameras == 1 && num_cameras > 1)
                        ? "  process: C1 only (S1 color SKU)"
                        : "");
        RCLCPP_INFO(this->get_logger(), "ISP mode: %s",
                    FastBalancedIspEnabled() ? "fast-balanced" : "quality-reference");
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
        // demosaiced BGR output for the Bayer camera profiles and stay
        // unused for MT9V034.
        cv::Mat left_image(profile.frame_height, profile.cam_width, CV_8U);
        cv::Mat right_image(profile.frame_height, profile.cam_width, CV_8U);
        cv::Mat left_front_image(profile.frame_height, profile.cam_width, CV_8U);
        cv::Mat right_front_image(profile.frame_height, profile.cam_width, CV_8U);
        cv::Mat left_color, right_color, left_front_color, right_front_color;
        cv::Mat half_raw[4], half_color[4];
        IspProcessor isp(CameraBgrIspMode()), half_isp(isp.mode());
        std::unique_ptr<SixCameraIspProcessor> six_isp;
        const bool use_six_scheduler = isp.UsesFastBalanced() &&
            !FastEnvTruthy("CYPERSTEREO_DISABLE_SIX_CAMERA_ISP");
        bool six_announced = false;

        while (is_running_ && rclcpp::ok())
        {
            if (!WaitForCameraFrame(frame_info, [this] { return is_running_ && rclcpp::ok(); })) break;

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

            cv_bridge::CvImage msg[6];
            int output_cameras = process_cameras;
            rclcpp::Time custom_time(image_timestamp*1e9); // seconds -> ns
            const bool half_res_lanes = is_bayer && process_cameras >= 4 &&
                IsSmartSensHalfResLaneLayout(six_image,
                                            static_cast<int>(hardware_version),
                                            static_cast<int>(software_version));

            if (is_bayer && half_res_lanes)
            {
                // S6/A6: lanes 2/3 are half images (V1.6: C3|C4 and C5|C6,
                // default board: C1|C2 and C3|C4).
                output_cameras = 6;
                const SixImageHalves halves = SelectSixImageHalves(
                    hardware_version, software_version, camera_gain.data(),
                    c56_gain.data());
                const int *half_order = SixImageHalfOutputOrder(hardware_version);
                if (!six_announced) {
                    const char *mode =
                        CameraModeName(static_cast<int>(hardware_version), true);
                    RCLCPP_INFO(this->get_logger(),
                        "%s six-image output: /cam0..5/image_raw = C1, C2 full, "
                        "C%dh C%dh C%dh C%dh half-width/half-height",
                        mode ? mode : "SmartSens", halves.camera[half_order[0]],
                        halves.camera[half_order[1]], halves.camera[half_order[2]],
                        halves.camera[half_order[3]]);
                    six_announced = true;
                }
                SplitAndVBinHalfResLanes(left_front_image, right_front_image, half_raw);
                const IspFrameJob jobs[] = {
                    {left_image, left_color, "cam1", camera_gain[0], SelectDeliveredPlaneBayer(hardware_version, software_version, 0), fpga_bayer_nr[0], raw_telemetry.RegionForPlane(0)},
                    {right_image, right_color, "cam2", camera_gain[1], SelectDeliveredPlaneBayer(hardware_version, software_version, 1), fpga_bayer_nr[1], raw_telemetry.RegionForPlane(1)},
                    {half_raw[0], half_color[0], halves.name[0], halves.gain[0], halves.bayer[0], false, {}, kHalfResNrGainScale},
                    {half_raw[1], half_color[1], halves.name[1], halves.gain[1], halves.bayer[1], false, {}, kHalfResNrGainScale},
                    {half_raw[2], half_color[2], halves.name[2], halves.gain[2], halves.bayer[2], false, {}, kHalfResNrGainScale},
                    {half_raw[3], half_color[3], halves.name[3], halves.gain[3], halves.bayer[3], false, {}, kHalfResNrGainScale}};
                if (use_six_scheduler) {
                    if (!six_isp) six_isp.reset(new SixCameraIspProcessor());
                    six_isp->Apply(jobs);
                } else {
                    isp.ApplyParallel(jobs, 2);
                    half_isp.ApplyParallel(jobs + 2, 4);
                }
                OrientSixImageHalves(hardware_version, half_raw, half_color);
                const cv::Mat *color[6] = {&left_color, &right_color,
                    &half_color[half_order[0]], &half_color[half_order[1]],
                    &half_color[half_order[2]], &half_color[half_order[3]]};
                for (int i = 0; i < 6; ++i) {
                    msg[i].encoding = "bgr8";
                    msg[i].image = *color[i];
                }
            }
            else if (is_bayer)
            {
                cv::Mat *raws[4] = {&left_image, &right_image, &left_front_image, &right_front_image};
                cv::Mat *colors[4] = {&left_color, &right_color, &left_front_color, &right_front_color};
                ApplyBayerCameraIsp(isp, raws, colors, process_cameras,
                    hardware_version, software_version, camera_gain, fpga_bayer_nr, raw_telemetry);
                msg[0].encoding = "bgr8";
                msg[0].image = left_color;
                if (process_cameras >= 2) {
                    msg[1].encoding = "bgr8";
                    msg[1].image = right_color;
                }
                if (process_cameras >= 4) {
                    // Planes arrive as C4, C3. V1.6 shows C3 then C4.
                    const bool c3_then_c4 = V16QuadShowsC3ThenC4(
                        static_cast<int>(hardware_version), false);
                    msg[2].encoding = "bgr8";
                    msg[2].image = c3_then_c4 ? right_front_color : left_front_color;
                    msg[3].encoding = "bgr8";
                    msg[3].image = c3_then_c4 ? left_front_color : right_front_color;
                }
            }
            else
            {
                // MT9V034 is monochrome: publish the two raw planes as mono8.
                msg[0].encoding = "mono8";
                msg[0].image = left_image;
                msg[1].encoding = "mono8";
                msg[1].image = right_image;
            }

            for (int i = 0; i < output_cameras; ++i) {
                msg[i].header.stamp = custom_time;
                msg[i].header.frame_id = "cam" + std::to_string(i);
            }

            RCLCPP_DEBUG(this->get_logger(), "image_timestamp %.6f", image_timestamp);
            
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
                RCLCPP_DEBUG(this->get_logger(), "imu_timestamp %.6f", imu_timestamp);

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

            for (int i = 0; i < output_cameras; ++i) {
                // Plane order is C1,C2,C4,C3. Six-image slots 2/3 are C3/C4,
                // so those temperatures sit at sensor indices 3 and 2.
                const int sensor_index = half_res_lanes && i >= 2 && i < 4 ? 5 - i : i;
                const bool valid = i < 4 ? sensors.temperature_valid[sensor_index]
                    : c56_valid && std::isfinite(c56_temperature[i - 4]);
                if (!valid) continue;
                sensor_msgs::msg::Temperature temperature;
                temperature.header.stamp = custom_time;
                temperature.header.frame_id = "cam" + std::to_string(i);
                temperature.temperature = i < 4 ? sensors.temperature[sensor_index]
                                                : c56_temperature[i - 4];
                temperature.variance = 0.0;
                temperature_pub[i]->publish(temperature);
            }
            for (int i = 0; i < output_cameras; ++i)
                cam_image_pub[i]->publish(*msg[i].toImageMsg());
            
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
