[English](README.md) | 简体中文

# 文档
使用指南与数据说明。
* 简体中文：[![](https://img.shields.io/badge/Download-HTML-blue.svg?style=flat)](https://readthedocs.org/projects/cyperstereo-sdk-docs-zh-cn/downloads/htmlzip/latest/) [![](https://img.shields.io/badge/Online-HTML-blue.svg?style=flat)](https://cyperstereo-sdk-docs-zh-cn.readthedocs.io/zh-cn/latest/)



# 1. Linux 编译
```c
cd ~
git clone https://github.com/Cyperstereo/CyperstereoSDK.git

cd ~/CyperstereoSDK/samples
mkdir build
cd build
cmake ..
make
```

# 2. Windows 编译
```c
编译 SDK 前请先自行安装 Visual Studio。

git clone https://github.com/Cyperstereo/CyperstereoSDK.git
cd ~/CyperstereoSDK/samples
mkdir build
cd build

cmake -G "Visual Studio 15 2017 Win64" ..
msbuild ALL_BUILD.vcxproj /property:Configuration=Release
```

# 3. ARM 编译
构建会自动识别 CPU 并启用 NEON/SIMD。通用 ARM 处理直接本机编译即可；在 RK3576 或 RK3588 上
本机编译也会识别出板子并使用硬件 ISP。也可以显式指定板子：
```c
cd ~/CyperstereoSDK
cmake -S samples -B samples/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTARGET_BOARD=rk3588
cmake --build samples/build --parallel 4
```

**交叉编译**（或强制指定调优目标）时传入 `-DTARGET_BOARD`：
```c
# --- Rockchip / 地平线 / 树莓派 ---
cmake -DTARGET_BOARD=rk3576 ..         # RK3576（Cortex-A72 + A53），默认使用硬件 ISP
cmake -DTARGET_BOARD=rk3588 ..         # RK3588（Cortex-A76 + A55），默认使用硬件 ISP
cmake -DTARGET_BOARD=sunrise-x5 ..     # 地平线 Sunrise/Journey X5（Cortex-A55）
cmake -DTARGET_BOARD=pi5 ..            # 树莓派 5（Cortex-A76）
cmake -DTARGET_BOARD=pi4 ..            # 树莓派 4（Cortex-A72）
cmake -DTARGET_BOARD=pi3 ..            # 树莓派 3（Cortex-A53）

# --- NVIDIA Jetson ---
cmake -DTARGET_BOARD=jetson-xavier ..  # Jetson AGX/NX Xavier（NVIDIA Carmel，ARMv8.2）
cmake -DTARGET_BOARD=jetson-orin ..    # Jetson AGX Orin / Orin NX / Orin Nano（Cortex-A78AE）
cmake -DTARGET_BOARD=jetson-thor ..    # Jetson Thor（Neoverse-V3AE，ARMv9）

# --- 兜底选项 ---
cmake -DTARGET_BOARD=native ..         # 针对编译主机 CPU 调优（-mcpu=native）
cmake -DTARGET_BOARD=generic ..        # 通用 ARMv8-A + NEON，不做特定 CPU 调优

make
```

## Rockchip 硬件 ISP

在 RK3576 和 RK3588 上，samples（`capture_image_imu`、`save_image_imu`）、ROS 和 ROS 2 默认使用硬件 ISP，六目同样适用。在板子上直接编译即可；交叉编译时传入 `-DTARGET_BOARD=rk3576` 或 `-DTARGET_BOARD=rk3588`。其他板子以及 x86 Linux、Windows、macOS 使用软件 ISP。

RK3576 不需要额外配置：

```c
cd ~/CyperstereoSDK
cmake -S samples -B samples/build -DCMAKE_BUILD_TYPE=Release -DTARGET_BOARD=rk3576
cmake --build samples/build --parallel 4
./samples/build/capture_image_imu
```

RK3588 的系统镜像在设备树里没有摄像头时默认关闭 ISP，需要启用一次并重启：

```c
sudo apt install device-tree-compiler           # 缺少 dtc 时安装
sudo ./scripts/setup_rk3588_hw_isp.sh           # 然后重启
sudo ./scripts/setup_rk3588_hw_isp.sh --remove  # 撤销，然后重启
```

脚本把 overlay 加到 `/boot/orangepiEnv.txt` 或 `/boot/armbianEnv.txt`。其他引导程序把 `--print` 打印的内容手动加进去。没有 overlay 时，程序会打印一次 `[rkisp] ...; using the software ISP from now on`，之后一直使用软件 ISP。`CYPERSTEREO_ISP_BACKEND=sw` 可在两块板子上强制使用软件 ISP。

ROS 和 ROS 2 同样：在板子上执行 `catkin_make` 或 `colcon build`。

## SmartSens 固件 2/3、2/4、2/5、2/6、2/7 和 2/8

Windows、Linux/V4L2、ROS、ROS 2 和 ARM 构建都使用 `src/usb/uvc/smartsens_metadata.h` 中
同一份带版本的元数据描述：

- 2/3：7 个 IMU 槽位，无 AE 遥测；
- 2/4：7 个 IMU 槽位，AE 遥测位于第 68..80 列；
- 2/5：13 个 IMU 槽位，AE 遥测位于第 122..134 列，16 Hz 相机触发对应 80 ms 的图像间隔阈值；
- 2/6：7 个 IMU 槽位和 2/4 的 AE 列，30 Hz 相机触发使用 HTS=452 的曝光换算
  （`29.761316872 us/line`）；
- 2/7（v0207）：在 2/6 布局基础上增加 USB 在线升级和 GNSS。IMU、第 68..80 列 AE 和 HTS=452
  行时间不变。GNSS 占第 81..105 列，只在该版本解析。接收机接 FPGA 管脚 L10，UART 115200 8N1。
  第 81 列为 `0x4700` 加上 GGA、GST 新鲜标志和 RMC 有效标志。`gnss.valid` 要求三者都新鲜、
  已定位、UTC 时间更新，且由 GST 推出的 hdop/vdop/pdop 都低于 50。图像字节从第 106 列继续；
- 2/8（v0208）：在 2/5 基础上增加 USB 在线升级，没有 GNSS。保留 2/5 的 13 个 IMU 槽位、
  第 122..134 列 AE、16 Hz 行时间、80 ms 图像间隔阈值和 Bayer 相位。AE 块之后的列不按 GNSS
  解析。

公开的 IMU 数组现在按 13 个样本定义。这改变了 C++ 对象布局，所以 Linux/ARM 部署必须彻底重新
编译 SDK 和所有使用方；不要把旧的 `Cyperlib` 或应用目标文件和新头文件混用。

版本/Bayer/列映射可以在没有 OpenCV 和相机的情况下检查：

```c
cmake -S tests -B tests/build -DCMAKE_BUILD_TYPE=Release
cmake --build tests/build --parallel
ctest --test-dir tests/build --output-on-failure
```

ARM 交叉编译时，可移植测试和 samples 构建都要传入目标平台的工具链。硬浮点与 softfp ABI 由
工具链决定；SDK 只选择 CPU/NEON 指令集：

```c
cmake -S tests -B tests/build-arm \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/arm-linux-toolchain.cmake
cmake --build tests/build-arm --parallel

cmake -S samples -B samples/build-arm \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/arm-linux-toolchain.cmake \
  -DTARGET_BOARD=rk3588
cmake --build samples/build-arm --parallel
```


# 4. ROS 编译
```c
cd ~/CyperstereoSDK/ros
catkin_make
source ./devel/setup.bash
rosrun CyperstereoRos capture_image_imu
```

# 5. ROS2 编译
```c
cd ~/CyperstereoSDK/ros2
source /opt/ros/humble/setup.sh
colcon build --symlink-install
source install/setup.sh
ros2 run data_cap capture_image_imu
```

# 6. 运行示例
```c
# 保存图像和 IMU 示例
mkdir left
mkdir right
mkdir imu
./save_image_imu

# 采集图像和 IMU 示例
./capture_image_imu
```

`capture_image_imu`、`save_image_imu` 以及 ROS 1 / ROS 2 节点默认输出 BGR888。在 RK3576 和
RK3588 上由硬件 ISP 生成。`./capture_image_imu --output-yuv422` 改为输出全范围 UYVY422
（`CV_8UC2`，字节顺序 `Cb,Y0,Cr,Y1`）。不要把它当作 limited 范围的 YUV 解码。RK3588 上示例
默认不打开预览窗口，需要时加 `--display`。

```c
./capture_image_imu --display        # 打开预览
./capture_image_imu --output-yuv422  # 输出 UYVY422 而不是 BGR888
./capture_image_imu --verbose        # 逐帧日志（RK3588 默认每秒一条）
```
