English | [简体中文](README.zh-CN.md)

# Documentations
guide doc and data spec.
* zh-Hans: [![](https://img.shields.io/badge/Download-HTML-blue.svg?style=flat)](https://readthedocs.org/projects/cyperstereo-sdk-docs-zh-cn/downloads/htmlzip/latest/) [![](https://img.shields.io/badge/Online-HTML-blue.svg?style=flat)](https://cyperstereo-sdk-docs-zh-cn.readthedocs.io/zh-cn/latest/)



# 1.Linux Compile
```c
cd ~
git clone https://github.com/Cyperstereo/CyperstereoSDK.git

cd ~/CyperstereoSDK/samples
mkdir build
cd build
cmake ..
make
```

# 2.Win Compile
```c
Before compiling the SDK, please ensure that you install Visual Studio on your own first.

git clone https://github.com/Cyperstereo/CyperstereoSDK.git
cd ~/CyperstereoSDK/samples
mkdir build
cd build

cmake -G "Visual Studio 15 2017 Win64" ..
msbuild ALL_BUILD.vcxproj /property:Configuration=Release
```

# 3.Arm Compile
The build auto-detects the CPU and enables NEON/SIMD. A native build is enough
for generic ARM processing, and a native build on an RK3576 or RK3588 also
recognizes the board and uses the hardware ISP. The board can also be named
explicitly:
```c
cd ~/CyperstereoSDK
cmake -S samples -B samples/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTARGET_BOARD=rk3588
cmake --build samples/build --parallel 4
```

When **cross-compiling** (or to force a specific tuning), pass `-DTARGET_BOARD`:
```c
# --- Rockchip / Horizon / Raspberry Pi ---
cmake -DTARGET_BOARD=rk3576 ..         # RK3576 (Cortex-A72 + A53), hardware ISP by default
cmake -DTARGET_BOARD=rk3588 ..         # RK3588 (Cortex-A76 + A55), hardware ISP by default
cmake -DTARGET_BOARD=sunrise-x5 ..     # Horizon Sunrise/Journey X5 (Cortex-A55)
cmake -DTARGET_BOARD=pi5 ..            # Raspberry Pi 5 (Cortex-A76)
cmake -DTARGET_BOARD=pi4 ..            # Raspberry Pi 4 (Cortex-A72)
cmake -DTARGET_BOARD=pi3 ..            # Raspberry Pi 3 (Cortex-A53)

# --- NVIDIA Jetson ---
cmake -DTARGET_BOARD=jetson-xavier ..  # Jetson AGX/NX Xavier (NVIDIA Carmel, ARMv8.2)
cmake -DTARGET_BOARD=jetson-orin ..    # Jetson AGX Orin / Orin NX / Orin Nano (Cortex-A78AE)
cmake -DTARGET_BOARD=jetson-thor ..    # Jetson Thor (Neoverse-V3AE, ARMv9)

# --- Fallbacks ---
cmake -DTARGET_BOARD=native ..         # tune for the build host CPU (-mcpu=native)
cmake -DTARGET_BOARD=generic ..        # portable ARMv8-A + NEON, no CPU-specific tuning

make
```

## Rockchip hardware ISP

On RK3576 and RK3588 the samples (`capture_image_imu`, `save_image_imu`), ROS and ROS 2 use the hardware ISP by default, six-image included. Build on the board, or pass `-DTARGET_BOARD=rk3576` or `-DTARGET_BOARD=rk3588` when cross-compiling. Other boards, and x86 Linux, Windows and macOS, stay on the software ISP.

RK3576 needs no extra setup:

```c
cd ~/CyperstereoSDK
cmake -S samples -B samples/build -DCMAKE_BUILD_TYPE=Release -DTARGET_BOARD=rk3576
cmake --build samples/build --parallel 4
./samples/build/capture_image_imu
```

RK3588 images leave the ISP off when the device tree describes no camera. Enable it once and reboot:

```c
sudo apt install device-tree-compiler           # dtc, if it is missing
sudo ./scripts/setup_rk3588_hw_isp.sh           # then reboot
sudo ./scripts/setup_rk3588_hw_isp.sh --remove  # undo, then reboot
```

The script adds an overlay to `/boot/orangepiEnv.txt` or `/boot/armbianEnv.txt`. For another boot loader, add the source printed by `--print`. Without the overlay the program prints `[rkisp] ...; using the software ISP from now on` once and stays on the software ISP. `CYPERSTEREO_ISP_BACKEND=sw` forces the software ISP on either board.

ROS and ROS 2 use the same builds: `catkin_make` or `colcon build` on the board.

## SmartSens firmware 2/3, 2/4, 2/5, 2/6, 2/7 and 2/8

Windows, Linux/V4L2, ROS, ROS 2 and ARM builds all use the same versioned
metadata description in `src/usb/uvc/smartsens_metadata.h`:

- 2/3: 7 IMU slots, no AE telemetry;
- 2/4: 7 IMU slots, AE telemetry in columns 68..80;
- 2/5: 13 IMU slots, AE telemetry in columns 122..134 and an 80 ms image-gap
  threshold for the 16 Hz camera trigger;
- 2/6: 7 IMU slots and the 2/4 AE columns, with HTS=452 exposure conversion
  (`29.761316872 us/line`) for the 30 Hz camera trigger;
- 2/7 (v0207): the 2/6 layout plus USB online upgrade and GNSS. IMU, AE
  columns 68..80 and the HTS=452 line time stay the same. GNSS occupies
  columns 81..105 and is parsed only for this version. The receiver is FPGA
  pin L10, UART 115200 8N1. Column 81 is `0x4700` plus fresh flags for GGA,
  GST and a valid RMC. `gnss.valid` requires all three fresh, a fix, a new
  UTC time, and GST-derived hdop/vdop/pdop below 50. Image bytes resume at
  column 106;
- 2/8 (v0208): USB online upgrade added to 2/5, with no GNSS. It keeps the
  2/5 13 IMU slots, AE columns 122..134, 16 Hz line time, 80 ms image-gap
  threshold and Bayer phase. Columns after the AE block are not read as
  GNSS.

The public IMU arrays are now sized for 13 samples. This changes the C++ object
layout, so Linux/ARM deployments must clean-rebuild the SDK and every consumer;
do not combine an old `Cyperlib` or application object with the new headers.

The version/Bayer/column mapping can be checked without OpenCV or a camera:

```c
cmake -S tests -B tests/build -DCMAKE_BUILD_TYPE=Release
cmake --build tests/build --parallel
ctest --test-dir tests/build --output-on-failure
```

For ARM cross-compilation, pass the deployment toolchain to both the portable
test and the samples build. The toolchain selects hard-float versus softfp ABI;
the SDK only selects the CPU/NEON instruction set:

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


# 4.ROS Compile
```c
cd ~/CyperstereoSDK/ros
catkin_make
source ./devel/setup.bash
rosrun CyperstereoRos capture_image_imu
```

# 5.ROS2 Compile
```c
cd ~/CyperstereoSDK/ros2
source /opt/ros/humble/setup.sh
colcon build --symlink-install
source install/setup.sh
ros2 run data_cap capture_image_imu
```

# 6.Run Samples
```c
#save image and imu samples  
mkdir left
mkdir right
mkdir imu
./save_image_imu

#capture image and imu samples  
./capture_image_imu
```

`capture_image_imu`, `save_image_imu` and the ROS 1 / ROS 2 nodes output
BGR888 by default. On RK3576 and RK3588 this comes from the hardware ISP.
`./capture_image_imu --output-yuv422` selects packed full-range UYVY422
instead (`CV_8UC2`, bytes `Cb,Y0,Cr,Y1`). Do not decode it as limited-range
YUV. On RK3588 the sample runs without a preview window unless `--display`
is given.

```c
./capture_image_imu --display        # show the preview
./capture_image_imu --output-yuv422  # UYVY422 instead of BGR888
./capture_image_imu --verbose        # log every frame (RK3588 defaults to once a second)
```
