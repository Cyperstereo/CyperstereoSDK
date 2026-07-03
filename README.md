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
The build auto-detects the CPU and enables NEON/SIMD. When you build **on the
device itself**, the default already picks the optimal `-mcpu=native`, so a
plain build is enough:
```c
cd ~/CyperstereoSDK/samples
mkdir build
cd build
cmake ..
make
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







