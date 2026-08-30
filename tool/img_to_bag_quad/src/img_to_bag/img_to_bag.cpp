#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <vector>
#include <dirent.h>
#include <stdlib.h>
#include <string>
#include <stdio.h>
#include <sensor_msgs/Imu.h>
#include <opencv2/opencv.hpp>

using namespace std;

const int IMAGE_WIDTH = 1280;
const int IMAGE_HEIGHT = 1024;
const int NUM_CAMERAS = 4;

int get_int_from_string(string& str);
vector<int> getFiles(char* dirc)
{
    vector<string> files;
    struct dirent *ptr;
    DIR *dir = opendir(dirc);

    if (dir == NULL)
    {
        perror("open dir error ...");
        exit(1);
    }

    while ((ptr = readdir(dir)) != NULL)
    {
        if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)
            continue;
        if (ptr->d_type == 8 || ptr->d_type == 4)
            files.push_back(ptr->d_name);
    }
    closedir(dir);

    vector<int> result;
    for (size_t i = 0; i < files.size(); i++)
        result.push_back(get_int_from_string(files[i]));
    sort(result.begin(), result.end());
    return result;
}

int get_int_from_string(string& str)
{
    int result = 0;
    for (size_t i = 0; i < str.size(); i++)
    {
        if (str[i] >= '0' && str[i] <= '9')
            result = result * 10 + str[i] - 48;
    }
    return result;
}

bool loadAndValidateImage(const string& filename, cv::Mat& image)
{
    image = cv::imread(filename);
    if (image.empty())
    {
        std::cout << "image empty: " << filename << std::endl;
        return false;
    }
    if (image.cols != IMAGE_WIDTH || image.rows != IMAGE_HEIGHT)
    {
        std::cout << "image size error, expected " << IMAGE_WIDTH << "x" << IMAGE_HEIGHT
                  << ", got " << image.cols << "x" << image.rows
                  << " in " << filename << std::endl;
        return false;
    }
    return true;
}

int main(int argc, char** argv)
{
    if (argc != 6)
    {
        cout << "usage: image_publisher cam0_folder cam1_folder cam2_folder cam3_folder imu_file" << endl;
        return -1;
    }

    ros::init(argc, argv, "image_imu_publisher");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);

    const char* cam_topics[NUM_CAMERAS] = {
        "/cam0/image_raw",
        "/cam1/image_raw",
        "/cam2/image_raw",
        "/cam3/image_raw"
    };

    image_transport::Publisher cam_pubs[NUM_CAMERAS];
    for (int c = 0; c < NUM_CAMERAS; c++)
        cam_pubs[c] = it.advertise(cam_topics[c], 1000);

    ros::Publisher imu_pub = nh.advertise<sensor_msgs::Imu>("imu0", 1000);

    const char* cam_folders[NUM_CAMERAS] = {argv[1], argv[2], argv[3], argv[4]};
    vector<int> image_counts[NUM_CAMERAS];
    for (int c = 0; c < NUM_CAMERAS; c++)
    {
        image_counts[c] = getFiles(const_cast<char*>(cam_folders[c]));
        std::cout << cam_folders[c] << " image count: " << image_counts[c].size() << std::endl;
    }
    std::cout << argv[5] << std::endl;

    FILE *fp = fopen(argv[5], "r");
    if (fp == NULL)
    {
        std::cout << "failed to open imu file" << std::endl;
        return -1;
    }

    double imu_time = 0.0;
    double last_imu_time = 0.0;
    float acceleration[3] = {0};
    float angular_v[3] = {0};

    ros::Rate loop_rate(25);
    int imu_seq = 1;
    int fscanf_return = fscanf(fp, "%lf,%f,%f,%f,%f,%f,%f",
                               &imu_time, angular_v, angular_v + 1, angular_v + 2,
                               acceleration, acceleration + 1, acceleration + 2);
    if (fscanf_return != 7)
    {
        std::cout << "imu format error" << std::endl;
        fclose(fp);
        return -1;
    }

    size_t frame_count = image_counts[0].size();
    for (int c = 1; c < NUM_CAMERAS; c++)
        frame_count = min(frame_count, image_counts[c].size());
    if (frame_count < 6)
    {
        std::cout << "not enough images in one or more camera folders" << std::endl;
        fclose(fp);
        return -1;
    }

    for (size_t i = 0; i < frame_count - 5; ++i)
    {
        if (!nh.ok())
            break;
        if (feof(fp))
            break;

        sensor_msgs::ImagePtr image_msgs[NUM_CAMERAS];
        int image_timestamp = image_counts[0][i];

        for (int c = 0; c < NUM_CAMERAS; c++)
        {
            if (image_counts[c][i] != image_timestamp)
            {
                std::cout << "camera timestamp mismatch at frame " << i
                          << ", cam" << c << " time: " << image_counts[c][i]
                          << ", cam0 time: " << image_timestamp << std::endl;
                fclose(fp);
                return -1;
            }

            std::string filename = string(cam_folders[c]) + "/" + to_string(image_counts[c][i]) + ".png";
            cv::Mat image;
            if (!loadAndValidateImage(filename, image))
            {
                fclose(fp);
                return -1;
            }

            image_msgs[c] = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
            image_msgs[c]->header.stamp = ros::Time(image_timestamp * 0.0001);
            image_msgs[c]->header.frame_id = "cam" + to_string(c);
        }

        if (i > 0 && (image_counts[0][i] - image_counts[0][i - 1]) > 5000)
            std::cout << "large interval in image" << std::endl;

        while (imu_time < image_counts[0][i + 1] * 0.0001)
        {
            if (last_imu_time > imu_time)
                std::cout << "imu time disorder" << last_imu_time << std::endl;
            if (imu_time - last_imu_time > 0.5)
                std::cout << "large interval in imu" << last_imu_time << std::endl;

            sensor_msgs::Imu imu_data;
            imu_data.header.stamp = ros::Time(imu_time);
            imu_data.header.frame_id = "body";
            imu_data.header.seq = imu_seq++;
            imu_data.linear_acceleration.x = acceleration[0];
            imu_data.linear_acceleration.y = acceleration[1];
            imu_data.linear_acceleration.z = acceleration[2];
            imu_data.angular_velocity.x = angular_v[0];
            imu_data.angular_velocity.y = angular_v[1];
            imu_data.angular_velocity.z = angular_v[2];

            if (imu_time != last_imu_time)
                imu_pub.publish(imu_data);

            last_imu_time = imu_time;
            fscanf_return = fscanf(fp, "%lf,%f,%f,%f,%f,%f,%f",
                                   &imu_time, angular_v, angular_v + 1, angular_v + 2,
                                   acceleration, acceleration + 1, acceleration + 2);
            if (fscanf_return != 7)
            {
                std::cout << "imu format error " << last_imu_time << std::endl;
                fclose(fp);
                return -1;
            }

            std::cout.setf(std::ios::fixed, std::ios::floatfield);
            std::cout.precision(6);
            std::cout << "imu_time: " << imu_time << " " << angular_v[0] << " " << angular_v[1] << " "
                      << angular_v[2] << " " << acceleration[0] << " " << acceleration[1] << " "
                      << acceleration[2] << std::endl;
        }

        std::cout << "image time: " << image_timestamp * 0.0001 << std::endl;
        for (int c = 0; c < NUM_CAMERAS; c++)
            cam_pubs[c].publish(image_msgs[c]);

        ros::spinOnce();
        loop_rate.sleep();
    }

    fclose(fp);
    return 0;
}
