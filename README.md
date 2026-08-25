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
for generic ARM processing, but RK3588 topology-aware scheduling (the four ISP
camera owners on CPU4-7 A76) requires `TARGET_BOARD=rk3588`:
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
cmake -DTARGET_BOARD=rk3588 ..         # RK3588 (Cortex-A76 + A55)
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

## SmartSens firmware 2/3, 2/4 and 2/5

Windows, Linux/V4L2, ROS, ROS 2 and ARM builds all use the same versioned
metadata description in `src/usb/uvc/smartsens_metadata.h`:

- 2/3: 7 IMU slots, no AE telemetry;
- 2/4: 7 IMU slots, AE telemetry in columns 68..80;
- 2/5: 13 IMU slots, AE telemetry in columns 122..134 and an 80 ms image-gap
  threshold for the 16 Hz camera trigger.

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

The fast-balanced capture path outputs packed UYVY422 by default on every
platform (`CV_8UC2`, byte order
`Cb,Y0,Cr,Y1`). It uses BT.601 full-range values and vertically duplicates the
ISP's 4:2:0 chroma rows. RGB tone mapping is deliberately deferred, so a
consumer that needs the former BGR888 appearance must perform a full-range
UYVY-to-BGR conversion and then apply the same tone mapping. Do not decode
this stream as studio/limited-range YUV. The quality-reference ISP continues
to output BGR888.

On RK3588, the four-camera sample is additionally headless by default and
restricts the complete capture process to the four Cortex-A76 cores CPU4-7
before any UVC, OpenCV, or ISP thread is created. The Cortex-A55 cores CPU0-3
are not used. The sample also disables LITTLE-core assistance and nested
per-frame ISP sharding.

Useful overrides are:

```c
# Show the local preview explicitly.
./capture_image_imu --display --isp-fast

# Restore the legacy BGR888 fast output (including RGB tone mapping).
./capture_image_imu --output-bgr --isp-fast

# Select packed full-range UYVY422 explicitly.
./capture_image_imu --output-yuv422 --isp-fast

# Restore per-frame metadata logging (RK3588 defaults to one sample/second).
./capture_image_imu --verbose
```

