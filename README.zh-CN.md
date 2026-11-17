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
构建会自动识别 CPU 并启用 NEON/SIMD。通用 ARM 处理直接本机编译即可；RK3588 的拓扑感知调度
（四个 ISP 相机线程固定在 CPU4-7 的 A76 上）需要 `TARGET_BOARD=rk3588`：
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
cmake -DTARGET_BOARD=rk3588 ..         # RK3588（Cortex-A76 + A55）
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

fast-balanced ISP（BGR888 和 UYVY422 输出）可以在 Rockchip ISP 硬核上运行，而不占用 CPU。
后端在编译时选择：

| 构建 | 默认 ISP |
|---|---|
| `TARGET_BOARD=rk3576`，或在 RK3576（AArch64）上本机 `auto` 构建 | ISP39 硬核（`src/usb/uvc/rk_hw_isp.cc`） |
| `TARGET_BOARD=rk3588` | 软件 ISP；硬件后端暂不支持 RK3588 的 ISP30 |
| 其他所有板子、x86 Linux、Windows、macOS | 软件 ISP |

以上同样适用于 samples（`capture_image_imu`、`save_image_imu`）、ROS 和 ROS 2：在 RK3576 上直接 `cmake`、`catkin_make` 或 `colcon build` 即默认使用硬核 ISP（含六目），交叉编译时传 `-DTARGET_BOARD=rk3576`；`CYPERSTEREO_ISP_BACKEND=sw` 对它们都可切回软件 ISP。

```c
cd ~/CyperstereoSDK
cmake -S samples -B samples/build -DCMAKE_BUILD_TYPE=Release -DTARGET_BOARD=rk3576
cmake --build samples/build --parallel 4
./samples/build/capture_image_imu                            # 硬件 ISP
CYPERSTEREO_ISP_BACKEND=sw ./samples/build/capture_image_imu # 强制使用软件 ISP
```

每帧 RAW8 从内存回灌进 ISP（rawrd -> ISP -> selfpath BGR / mainpath UYVY），不经过 rkaiq。
黑电平、白平衡、数字曝光和高光压缩与同一帧的软件 ISP 一致；YNR/CNR 强度随传感器增益变化。
共享亮度的色调映射仍在 CPU 上，融合在从 ISP 缓冲区拷出的那一步里。低增益下输出与软件 ISP
的 PSNR 在 40-50 dB。

硬件后端需要两路空闲的 `rkisp-vir` 虚拟 ISP（每种 CFA 相位一路），以及对其 `/dev/video*`
和 `/dev/v4l-subdev*` 节点的读写权限（通常是 `video` 组）。被其他程序占用的虚拟 ISP（例如等待
传感器的 `rkaiq_3A`）会被跳过，所以 `rkaiq_3A` 可以继续运行。在一块新的 RK3576 板上不需要
任何额外操作：克隆、按上面的步骤编译、运行即可。以下情况一批图像会退回软件 ISP：

- 打不开任何虚拟 ISP，或 ISP 报错。这时会打印一次
  `[rkisp] ...; using the software ISP from now on`，之后该进程一直使用软件 ISP；
- 这批图像高度或输出格式不同、全图（或半图）宽度不同，或宽高为奇数。

ISP 没有软件 BayerNR 的替代模块。需要 BayerNR 的图像（NR 增益不低于
`CYPERSTEREO_RKISP_MAX_NR_GAIN`，默认 3，且 FPGA BayerNR 关闭）会先在 CPU 上做软件 BayerNR，
与白平衡融合，再由 ISP 在这份线性 RAW 上跑剩余流程。

SC136HGS 六目（S6 序列号）的帧通过 `IspProcessor::ApplyParallelWith` 作为一个硬件批次运行：
C1/C2 全尺寸，加上四张 `(cam_width/2) x frame_height` 的半图。半图的白平衡在 CPU 上完成，
因此两张半图左右并排放进同一个 ISP 帧，中间隔 32 列镜像填充。ISP 以全高度去马赛克，CPU 在
色调映射前把线性行 2:1 平均。另一种 CFA 相位的半图向内错开一行一列放置，这样 NR 增益相近的
半图可以共用一帧，帧数也能在两路虚拟 ISP 间均分（仅 BGR 输出；UYVY 让每对半图保持同一相位）。
ISP 保持六目尺寸打开（1280x1024 相机为 1312x1026），全图填充到这个尺寸。

AR0234 六目（A6 序列号）走同样的批次。它的 960x1080 半图无间隔地填满 1920 宽的 selfpath
（mainpath 上的 UYVY 保留 32 列间隔），每种 CFA 相位各两帧，不需要错一像素。一个批次超过
`CYPERSTEREO_RKISP_HW_BUDGET_MP`（6.4 MP 的 ISP 帧，约 24 ms 核时间）时，多出的帧交给软件
ISP，优先拿半图对，在 CPU6/7 上与 ISP 核并行运行：AR0234 六目的四个 1920x1080 帧中三个留在
ISP 上，AR0234 四目的四张图中三张留在 ISP 上。SC136HGS 的批次低于预算。没有硬件 ISP 时，
AR0234 六目使用六线程软件调度（`SixCameraIspProcessor`）。

运行时调节：

| 变量 | 默认值 | 含义 |
|---|---|---|
| `CYPERSTEREO_ISP_BACKEND` | hardware | `sw`（或 `software`、`cpu`）让所有批次都走软件 ISP |
| `CYPERSTEREO_RKISP_MAX_NR_GAIN` | 3 | 从这个 NR 增益起，没有 FPGA BayerNR 的图像先在 CPU 上做软件 BayerNR 再进 ISP |
| `CYPERSTEREO_RKISP_NR_GAIN_SCALE` | 0.4 | YNR/CNR 噪声模型增益相对 NR 增益的比例；越大平坦墙面越平滑，但细纹理损失越多 |
| `CYPERSTEREO_RKISP_PACK_GAP` | 32 | 同一 ISP 帧中两张六目半图之间的镜像列数（4 的倍数） |
| `CYPERSTEREO_RKISP_PACK` | 1 | `0` 让每张六目半图单独占一个 ISP 帧 |
| `CYPERSTEREO_RKISP_CORE_MHZ` | 594 | ISP 核时钟；`0` 保持驱动的 396 MHz，`702` 选最高档 |
| `CYPERSTEREO_RKISP_HW_BUDGET_MP` | 6.4 | 每批 ISP 帧的百万像素上限；超出部分由软件 ISP 并行处理（`0` = 不限制） |

在 RK3576 上测得，四个 1280x1024 相机 30 fps，`performance` 调频，ISP 核 396 MHz
（默认的 594 MHz 下，相同帧的离线回放从 20.3-21 ms 降到 17.0-17.4 ms）：

| | 软件 ISP | 硬件 ISP |
|---|---|---|
| 每个四目帧的 CPU 时间 | 52-56 ms（约 1.6-1.7 核） | 19-20 ms（约 0.6 核） |
| 每个四目帧的 ISP 延迟 | 14-15 ms | 21-22 ms |

硬件 ISP 延迟更高，是因为两路虚拟 ISP 共用一个 ISP 核、依次运行。它为应用腾出了大约一个 A72 核。

SC136HGS 六目（C1/C2 1280x1024 加四张 640x512 半图），条件相同（594 MHz 下离线回放从
20.2 ms 降到 17.0-17.5 ms）：

| | 软件 ISP | 硬件 ISP |
|---|---|---|
| 每个六目帧的 CPU 时间，离线回放 | 46-55 ms（约 1.4-1.7 核） | 21-25 ms（约 0.65-0.75 核） |
| 每个六目帧的 CPU 时间，实机 | 48-49 ms（约 1.45 核） | 25-27 ms（约 0.75-0.8 核） |
| 每个六目帧的 ISP 延迟，实机 | 20 ms | 22.5-23 ms |

AR0234（1920x1080），条件相同，硬件 ISP 为默认的 594 MHz（括号内为 396 MHz）。四目一行用
两路真实图像各复制一份回放；手头只有 AR0234 六目相机：

| | 软件 ISP | 硬件 ISP |
|---|---|---|
| 六目 BGR，实机：延迟 / 每帧 CPU | 28.3 ms / 101 ms（3.0 核） | 24.1-24.7 ms（30.1-30.4 ms）/ 55 ms（1.65 核） |
| 六目 UYVY，实机：延迟 / 每帧 CPU | 46.5 ms（20 fps）/ 108 ms | 23.8-24.2 ms（30.5 ms）/ 64 ms（1.9 核） |
| 六目 BGR，离线：延迟 / CPU | 30.8 ms / 77 ms | 21.8 ms（25.7 ms）/ 40 ms |
| 四目 BGR，离线：延迟 / CPU | 20.5 ms / 81 ms | 23.0 ms（25.6 ms）/ 37 ms |

396 MHz 且不设预算（`CYPERSTEREO_RKISP_HW_BUDGET_MP=0`）时，四个 1920x1080 帧离线需要 31 ms
核时间，实机 34 ms，达不到 30 fps。

ISP 核默认运行在 594 MHz；单靠驱动，这些帧尺寸只会得到 396 MHz。rawrd 输入时，Rockchip 驱动
按正在出流的虚拟 ISP 的输入尺寸选核时钟：(w x h 之和) x 30 fps x (1 + 每路 25%)，对照
400/400/500/600/702 MHz 档位，并且只在 `RKISP_CMD_MULTI_DEV_FORCE_ENUM` 时才重新读取这些尺寸。
后端在第一帧之前先把每路虚拟 ISP 设成较大的输入尺寸并发出该命令，然后再按真实尺寸打开自己的
两路虚拟 ISP，于是驱动选中 600 MHz 档（GPLL 分频得到 594 MHz）。这不需要改内核、不需要 root，
也不占用额外的虚拟 ISP，`rkaiq_3A` 运行时同样有效；如果已有别的程序在用 ISP 出流，尺寸无法
重新读取，会打印一条提示并保持驱动的时钟。输出与 396 MHz 逐位一致，CPU 时间不变；进程退出后
时钟恢复为驱动的选择。`CYPERSTEREO_RKISP_CORE_MHZ=0` 保持驱动的时钟，`702` 选最高档，对这些
批次不会更快。

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

fast-balanced 采集路径在所有平台上默认输出紧凑排列的 UYVY422（`CV_8UC2`，字节顺序
`Cb,Y0,Cr,Y1`）。它使用 BT.601 全范围数值，并把 ISP 的 4:2:0 色度行在垂直方向上复制。RGB
色调映射被有意推迟，所以需要原先 BGR888 观感的使用方，必须先做全范围 UYVY 到 BGR 的转换，
再应用同样的色调映射。不要把这路流当作 studio/limited 范围的 YUV 解码。质量参考 ISP 仍输出
BGR888。

在 RK3588 上，四目示例还默认无界面运行，并在创建任何 UVC、OpenCV 或 ISP 线程之前，把整个采集
进程限制在 CPU4-7 这四个 Cortex-A76 核上，不使用 CPU0-3 的 Cortex-A55 核。示例还关闭了小核
辅助和单帧内嵌套的 ISP 分片。

常用的覆盖选项：

```c
# 显式打开本地预览。
./capture_image_imu --display --isp-fast

# 恢复旧的 BGR888 快速输出（包括 RGB 色调映射）。
./capture_image_imu --output-bgr --isp-fast

# 显式选择紧凑的全范围 UYVY422。
./capture_image_imu --output-yuv422 --isp-fast

# 恢复逐帧元数据日志（RK3588 默认每秒一条）。
./capture_image_imu --verbose
```
