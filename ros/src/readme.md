```bash
cd ~/ar0234_ws/CyperstereoSDK/ros
catkin_make -DCMAKE_BUILD_TYPE=Release
source ./devel/setup.bash
rosrun CyperstereoRos capture_image_imu
```

AR0234 六路固件启用与 samples/save_image_imu 相同的六路 ISP：
四个大核处理 C1、C2、C3+C4、C5+C6，RK3576 的两个小核辅助大图前端和输出。
本机构建自动识别 RK3576；交叉编译可显式传入 `-DTARGET_BOARD=rk3576`。
保留每路独立白平衡、数字曝光、Bayer 相位和降噪参数，输出仍为 `bgr8`。

在当前 1920×1080 AR0234 六路固件下：

| ROS 话题 | 相机 | 输出尺寸 |
|---|---|---|
| `/cam0/image_raw` | C1 | 1920×1080 |
| `/cam1/image_raw` | C2 | 1920×1080 |
| `/cam2/image_raw` | C3 | 960×540 |
| `/cam3/image_raw` | C4 | 960×540 |
| `/cam4/image_raw` | C5 | 960×540 |
| `/cam5/image_raw` | C6 | 960×540 |

六路使用同一个硬件图像时间戳；C1/C2 保持 SDK 的 180° 旋转，C5 与 capture/save 一样做垂直翻转。
`/cam2` 和 `/cam3` 在该固件下由原来的拼接平面改为独立 C3/C4，订阅端应按上表更新尺寸及对应关系。
温度话题 `/cam0/temperature` 至 `/cam5/temperature` 使用对应相机的有效遥测；IMU 保持 `imu0`。
非 AR0234 六路固件继续使用原有相机数量和处理路径。

ISP 直接写入可复用的 ROS 消息缓冲；只有所有订阅者都释放后才复用，避免覆盖仍在发送的图像。
capture/ROS 本来就用 RAW 缓冲交换，因此无需复制 save 程序专用的保存队列/RAW 回收池。

保留对照开关：`CYPERSTEREO_DISABLE_SIX_CAMERA_ISP=1 CYPERSTEREO_FAST_BIG_LITTLE=0` 恢复两批 ISP 调度，六路话题映射不变。
只关闭小核协作可设置 `CYPERSTEREO_FAST_BIG_LITTLE=0`。

构建前请先加载已安装的 ROS1 环境（setup.bash），再运行上述 catkin_make 命令。
