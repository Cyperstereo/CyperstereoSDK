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
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
using namespace std;

int get_int_from_string(string& str);
//get num in file name in sort from small to big
vector<int> getFiles(char* dirc){
    vector<string> files;
    struct dirent *ptr;
    DIR *dir;
    dir = opendir(dirc);
    
    if(dir == NULL)
    {
        perror("open dir error ...");
        exit(1);
    }

    while((ptr = readdir(dir)) != NULL){
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir  
            continue;  
        if(ptr->d_type == 8)//it;s file
        {
            files.push_back(ptr->d_name);
        }

        else if(ptr->d_type == 10)//link file
            continue;
        else if(ptr->d_type == 4) //dir
        {
            files.push_back(ptr->d_name);
        }
    }
    closedir(dir);
    
    vector<int> result;
    for(int i=0;i < files.size();i++)
    {
        result.push_back(get_int_from_string(files[i]));
    }
    sort(result.begin(),result.end());

    for(size_t i = 0; i < result.size();++i){
        //cout << result[i] << endl;
    }
    return result;
}

int get_int_from_string(string& str)
{
    int result = 0;
    for(int i = 0; i < str.size(); i++)
    {
        if (str[i] >= '0'&& str[i] <= '9')  
        {  
            result = result * 10 + str[i] - 48;  
        }  
    }
    return result;
}


int main(int argc, char** argv)
{   
    if (argc != 4)
    {
        cout << "usage: image_publisher left_image_folder right_image_folder imu_file " << endl;
        return -1;
    }

    ros::init(argc, argv, "image_imu_to_rosbag");//启动节点，设置名称
    ros::NodeHandle nh;//设置节点进程句柄
    image_transport::ImageTransport it(nh);
    image_transport::Publisher cam0_image_pub = it.advertise("/cam0/image_raw", 1000);//消息名称，缓冲区大小
    image_transport::Publisher cam1_image_pub = it.advertise("/cam1/image_raw", 1000);
    ros::Publisher IMU_pub = nh.advertise<sensor_msgs::Imu>("imu0", 1000); 
    std::cout << argv[1] << std::endl;
    std::cout << argv[2] << std::endl;
    std::cout << argv[3] << std::endl;
    vector<int> left_image_count = getFiles(argv[1]);// 
    vector<int> right_image_count = getFiles(argv[2]);
    FILE *fp;
    fp = fopen(argv[3],"r");
    double imu_time,last_imu_time;
    float acceleration[3],angular_v[3];

    //the argv is the url of the image,may we can use that for all images
    ros::Rate loop_rate(25);
    int time_count_left,time_count_right;
    int imu_seq = 1;
    std::map<int,int> imu_big_interval;
    int fscanf_return;
    fscanf_return = fscanf(fp,"%lf,%f,%f,%f,%f,%f,%f",
                &imu_time,angular_v,angular_v+1,angular_v+2,acceleration,acceleration+1,acceleration+2);
    if (fscanf_return != 7)
    {
        std::cout << "imu format error " << last_imu_time <<std::endl;
        fclose(fp);
        return -1;
    }
    for(size_t i = 10; (i < left_image_count.size() - 10)&&(i < right_image_count.size() - 10) ;++i)
    {
        if(!nh.ok())
            break;

        if (feof(fp))
            break;
        
        ostringstream stringStream;
        //转换左图
        std::string left_filename = string(argv[1]) + "/" + to_string(left_image_count[i]) + ".png";
        cv::Mat left_image = cv::imread(left_filename);
        if(left_image.empty())
        {
            std::cout << "left image empty" << std::endl;
            return 0;
        }
        sensor_msgs::ImagePtr msg0 = cv_bridge::CvImage(std_msgs::Header(), "bgr8", left_image).toImageMsg();
        time_count_left = left_image_count[i];
        msg0->header.stamp = ros::Time(time_count_left * 0.0001);
    
        //转换右图
        std::string right_filename = string(argv[2])+"/"+ to_string(right_image_count[i]) + ".png";
        cv::Mat right_image = cv::imread(right_filename);
        if(right_image.empty())
        {
            std::cout << "right image empty" << std::endl;
            return 0;
        }
        sensor_msgs::ImagePtr msg1 = cv_bridge::CvImage(std_msgs::Header(), "bgr8", right_image).toImageMsg();
        time_count_right = right_image_count[i];
        msg1->header.stamp = ros::Time(time_count_left * 0.0001);
        if(time_count_left != time_count_right)
        {
           std::cout << "left image time != right image time" <<  time_count_left << std::endl;
           return -1; 
        }
        if((left_image_count[i] - left_image_count[i-1]) > 5000)
        {
            std::cout << "large interval in image" << std::endl;
        }
        while (imu_time < left_image_count[i+1]*0.0001)
        {
            if(last_imu_time > imu_time)
            {
                std::cout << "imu time disorder" << last_imu_time << std::endl;
            }
            if ( imu_time - last_imu_time > 0.5)
            {
                std::cout << "large interval in imu" << last_imu_time << std::endl;
            }
            
            sensor_msgs::Imu imu_data;
            imu_data.header.stamp = ros::Time(imu_time);
            imu_data.header.frame_id = "body";
            imu_data.header.seq = imu_seq++;
            //acc  
            imu_data.linear_acceleration.x = acceleration[0]; 
            imu_data.linear_acceleration.y = acceleration[1];
            imu_data.linear_acceleration.z = acceleration[2];
           
            //gyro
            imu_data.angular_velocity.x = angular_v[0]; 
            imu_data.angular_velocity.y = angular_v[1]; 
            imu_data.angular_velocity.z = angular_v[2];
            
            if(imu_time != last_imu_time)
            {
                IMU_pub.publish(imu_data);
            }                
            last_imu_time = imu_time;
            fscanf_return = fscanf(fp,"%lf,%f,%f,%f,%f,%f,%f",
                &imu_time,angular_v,angular_v+1,angular_v+2,acceleration,acceleration+1,acceleration+2);
            if (fscanf_return != 7)
            {
                std::cout << "imu format error " << last_imu_time <<std::endl;
                fclose(fp);
                return -1;
            }
            std::cout.setf(std::ios::fixed, std::ios::floatfield);
	        std::cout.precision(6);
            std::cout << "imu_time: " << imu_time << " " << angular_v[0] << " " << angular_v[1] << " " << angular_v[2] << " " << acceleration[0] <<  " " << acceleration[1] <<  " " << acceleration[2] << std::endl;
        }
        std::cout << "image time: " << time_count_left * 0.0001 << std::endl;
        cam0_image_pub.publish(msg0);//发布左图
        cam1_image_pub.publish(msg1);
        ros::spinOnce();
        loop_rate.sleep();
    }
    
    fclose(fp);
    return 0;

}
