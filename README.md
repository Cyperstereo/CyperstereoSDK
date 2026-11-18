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
recognizes the board (on RK3588: the four ISP camera owners on CPU4-7 A76 and
the hardware ISP). The board can also be named explicitly:
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

The fast-balanced ISP (BGR888 and UYVY422 output) can run on the Rockchip
ISP core instead of the CPU. The backend is selected at build time:

| Build | Default ISP |
|---|---|
| `TARGET_BOARD=rk3576`, or a native `auto` build on an RK3576 (AArch64) | hardware ISP39 core (`src/usb/uvc/rk_hw_isp.cc`) |
| `TARGET_BOARD=rk3588`, or a native `auto` build on an RK3588 | hardware ISP30 cores (same file), once enabled with `scripts/setup_rk3588_hw_isp.sh` (see [RK3588](#rk3588-isp30)) |
| every other board, x86 Linux, Windows, macOS | software ISP |

This applies to the samples (`capture_image_imu`, `save_image_imu`), ROS and
ROS 2 alike, six-image mode included: a plain `cmake`, `catkin_make` or
`colcon build` on an RK3576 or RK3588 uses the hardware ISP, a cross build
passes `-DTARGET_BOARD=rk3576` or `-DTARGET_BOARD=rk3588`, and
`CYPERSTEREO_ISP_BACKEND=sw` switches any of them back to the software ISP.
The rest of this section describes the RK3576; the RK3588 differences are in
[RK3588](#rk3588-isp30).

```c
cd ~/CyperstereoSDK
cmake -S samples -B samples/build -DCMAKE_BUILD_TYPE=Release -DTARGET_BOARD=rk3576
cmake --build samples/build --parallel 4
./samples/build/capture_image_imu                            # hardware ISP
CYPERSTEREO_ISP_BACKEND=sw ./samples/build/capture_image_imu # force the software ISP
```

Each RAW8 frame is replayed from memory through the ISP (rawrd -> ISP ->
selfpath BGR / mainpath UYVY) without rkaiq. Black level, white balance,
digital exposure and the highlight roll-off follow the software ISP of the
same frame; YNR/CNR strength follows the sensor gain. The shared-luminance
tone stays on the CPU, fused into the copy out of the ISP buffer. At low gain
the output is within 40-50 dB PSNR of the software ISP.

The hardware backend needs two free `rkisp-vir` virtual ISPs (one per CFA
phase) and read/write access to their `/dev/video*` and `/dev/v4l-subdev*`
nodes (normally the `video` group). A virtual ISP held by another client, for
example `rkaiq_3A` waiting for a sensor, is skipped, so `rkaiq_3A` can keep
running. Nothing else is needed on a new RK3576 board: clone, build with the
steps above and run. A batch falls back to the software ISP when:

- no virtual ISP can be opened or the ISP reports an error. This prints
  `[rkisp] ...; using the software ISP from now on` once, and the process
  then stays on the software ISP;
- its images differ in height or output format, its full images (or its
  half images) differ in width, or a width or height is odd.

The ISP has no stand-in for the software BayerNR. An image that needs it (NR
gain at least `CYPERSTEREO_RKISP_MAX_NR_GAIN`, default 3, and the FPGA
BayerNR off) gets the software BayerNR on the CPU, fused with the white
balance, and the ISP then runs the rest of the pipeline on that linear RAW.

SC136HGS six-image (S6 serial) frames run as one hardware batch through
`IspProcessor::ApplyParallelWith`: C1/C2 at full size and the four
`(cam_width/2) x frame_height` halves. The halves get their white balance on
the CPU, so two of them share one ISP frame side by side, 32 mirrored columns
apart. The ISP demosaics them at full height and the CPU averages the linear
rows 2:1 before the tone. A half of the other CFA phase is placed one row and
one column in, which lets halves of similar NR gain share a frame and splits
the frames evenly between the two virtual ISPs (BGR output only; UYVY keeps
each pair within one phase). The ISP stays open at the six-image size
(1312x1026 for 1280x1024 cameras), and the full images are padded to it.

AR0234 six-image (A6 serial) frames take the same batch. Their 960x1080
halves fill the 1920-wide selfpath with no gap (UYVY on the mainpath keeps
the 32 columns), and each CFA phase gets two frames without the one-pixel
shift. A batch above `CYPERSTEREO_RKISP_HW_BUDGET_MP` (6.4 MP of ISP frames,
about 24 ms of core time) leaves frames to the software ISP, half pairs
first, which runs them on CPU6/7 while the core works: AR0234 six-image
keeps three of its four 1920x1080 frames on the ISP, and AR0234 quad three
of its four images. SC136HGS batches stay below the budget. Without the
hardware ISP, AR0234 six-image frames use the six-thread software scheduler
(`SixCameraIspProcessor`).

Runtime tuning:

| Variable | Default | Meaning |
|---|---|---|
| `CYPERSTEREO_ISP_BACKEND` | hardware | `sw` (or `software`, `cpu`) keeps every batch on the software ISP |
| `CYPERSTEREO_RKISP_MAX_NR_GAIN` | 3 | NR gain from which images without FPGA BayerNR get the software BayerNR on the CPU before the ISP |
| `CYPERSTEREO_RKISP_NR_GAIN_SCALE` | 0.4 | YNR/CNR noise-model gain relative to the NR gain; higher values smooth flat walls more but lose fine texture |
| `CYPERSTEREO_RKISP_PACK_GAP` | 32 | Mirrored columns between two six-image halves in one ISP frame (multiple of 4) |
| `CYPERSTEREO_RKISP_PACK` | 1 | `0` gives every six-image half its own ISP frame |
| `CYPERSTEREO_RKISP_CORE_MHZ` | 594 | ISP core clock; `0` keeps the driver's 396 MHz, `702` selects the top step |
| `CYPERSTEREO_RKISP_HW_BUDGET_MP` | 6.4 | ISP frame megapixels per batch; frames beyond it run on the software ISP concurrently (`0` = no limit) |

Measured on RK3576 with four 1280x1024 cameras at 30 fps, `performance`
governor, ISP core at 396 MHz (at the default 594 MHz the offline replay of
the same frames takes 17.0-17.4 ms instead of 20.3-21 ms):

| | Software ISP | Hardware ISP |
|---|---|---|
| CPU time per four-camera frame | 52-56 ms (about 1.6-1.7 cores) | 19-20 ms (about 0.6 cores) |
| ISP latency per four-camera frame | 14-15 ms | 21-22 ms |

The hardware ISP has the higher latency because both virtual ISPs share one
ISP core and run back to back. It frees about one A72 core for the
application.

SC136HGS six-image (C1/C2 1280x1024 plus four 640x512 halves), same
conditions (at 594 MHz the offline replay takes 17.0-17.5 ms instead of
20.2 ms):

| | Software ISP | Hardware ISP |
|---|---|---|
| CPU time per six-image frame, offline replay | 46-55 ms (about 1.4-1.7 cores) | 21-25 ms (about 0.65-0.75 cores) |
| CPU time per six-image frame, live | 48-49 ms (about 1.45 cores) | 25-27 ms (about 0.75-0.8 cores) |
| ISP latency per six-image frame, live | 20 ms | 22.5-23 ms |

AR0234 (1920x1080), same conditions, hardware ISP at the default 594 MHz
(396 MHz in brackets). The quad row replays two real planes twice; the
six-image unit was the only AR0234 camera available:

| | Software ISP | Hardware ISP |
|---|---|---|
| Six-image BGR, live: latency / CPU per frame | 28.3 ms / 101 ms (3.0 cores) | 24.1-24.7 ms (30.1-30.4 ms) / 55 ms (1.65 cores) |
| Six-image UYVY, live: latency / CPU per frame | 46.5 ms (20 fps) / 108 ms | 23.8-24.2 ms (30.5 ms) / 64 ms (1.9 cores) |
| Six-image BGR, offline: latency / CPU | 30.8 ms / 77 ms | 21.8 ms (25.7 ms) / 40 ms |
| Quad BGR, offline: latency / CPU | 20.5 ms / 81 ms | 23.0 ms (25.6 ms) / 37 ms |

At 396 MHz with no budget (`CYPERSTEREO_RKISP_HW_BUDGET_MP=0`) the four
1920x1080 frames take 31 ms of core time offline and 34 ms live, which misses
30 fps.

The ISP core runs at 594 MHz by default; the driver alone would clock it at
396 MHz for these frame sizes. With rawrd input the Rockchip driver picks the
core clock from the input sizes of the streaming virtual ISPs, (sum of
w x h) x 30 fps x (1 + 25% per ISP) against its 400/400/500/600/702 MHz
steps, and it re-reads those sizes only on `RKISP_CMD_MULTI_DEV_FORCE_ENUM`.
Before its first frame the backend gives every virtual ISP a large input size
and issues that command, then opens its two virtual ISPs at their real size,
so the driver selects the 600 MHz step (594 MHz from GPLL). This needs no
kernel change, no root and no extra virtual ISP, and works with `rkaiq_3A`
running; when another client is already streaming an ISP the sizes cannot be
re-read, a message is printed and the driver's clock stays. The output is
bit-identical to 396 MHz and the CPU time is unchanged; the clock returns to
the driver's choice when the process exits. `CYPERSTEREO_RKISP_CORE_MHZ=0`
keeps the driver's clock, `702` selects the top step, which is no faster for
these batches.

### RK3588 (ISP30)

The RK3588 has two ISP30 cores. Most RK3588 images ship their virtual ISPs
disabled when no camera sensor is described in the device tree, so enable them
once and reboot:

```c
sudo apt install device-tree-compiler           # dtc, if missing
sudo ./scripts/setup_rk3588_hw_isp.sh           # install the overlay, then reboot
sudo ./scripts/setup_rk3588_hw_isp.sh --remove  # undo, then reboot
sudo ./scripts/setup_rk3588_hw_isp.sh --print   # show the overlay source
```

The script compiles a device-tree overlay that enables both ISP cores, their
IOMMUs and three virtual ISPs per core (`rkisp0-vir0..2`, `rkisp1-vir0..2`),
installs it as `/boot/overlay-user/cyperstereo-rkisp.dtbo` and adds it to
`user_overlays=` in `/boot/orangepiEnv.txt` (or `/boot/armbianEnv.txt`); other
boot loaders need the `--print` output added by hand. It was tested on an
Orange Pi 5 Plus (Ubuntu 22.04, kernel 6.1.43, rkisp driver v2.4.0). Without
the overlay the backend prints `[rkisp] no usable virtual ISP: no rkisp-vir
media device; using the software ISP from now on` once.

Differences from the RK3576:

- Four virtual ISPs are used, one per CFA phase on each core, and the frames
  of a phase alternate between the cores, so the two cores run concurrently.
  The driver's clock estimate (above) counts only streaming virtual ISPs, so
  each core needs two of them for the 600 MHz step: before the first frame the
  backend sets the first virtual ISP of each core to 3840x2160 and the others
  to 2560x1536 (larger inputs make the driver read every frame back twice),
  which selects 594 MHz on both cores. `CYPERSTEREO_RKISP_CORE_MHZ=0` keeps
  the driver's 300 MHz; `702` behaves like the default.
- The selfpath has no RGB output, so BGR output comes from the mainpath UYVY
  and is converted to BGR on the CPU during the copy out, before the tone.
- ISP frames are at most 2560 wide; wider batches run on the software ISP.
  `CYPERSTEREO_RKISP_HW_BUDGET_MP` applies per core.
- The ISP30 takes the RK3576 DPCC and colour-space configs and the debayer of
  the imx415 ISO-50 tuning shipped in `/etc/iqfiles`. It runs no YNR, CNR or
  sharpening (the ISP30 noise filters differ from the ISP39's and have no
  captured tuning yet), so at high gain flat areas keep more grain than the
  software ISP. With four SC136HGS cameras at gain 10 the output is within
  34.7-36.9 dB PSNR of the software ISP, BGR and UYVY alike.

Measured on the Orange Pi 5 Plus with four SC136HGS 1280x1024 cameras at
30 fps (gain 10), `performance` governor, ISP cores at 594 MHz. Live numbers
are `capture_image_imu` (`isp_wall`, process CPU); offline numbers are
`isp_quad_bench` on frames from the same camera:

| | Software ISP | Hardware ISP |
|---|---|---|
| BGR (default), live: ISP latency / CPU | 8.05 ms / 0.99 cores | 9.36 ms / 0.62 cores |
| UYVY, live: ISP latency / CPU | 14.8 ms / 1.68 cores | 8.5 ms / 0.54 cores |
| BGR, offline: latency / CPU per frame | 7.9 ms / 30.7 ms | 9.5 ms / 17.4 ms |
| UYVY, offline: latency / CPU per frame | 15.1-18.6 ms / 59-72 ms | 8.5 ms / 13.4 ms |

Each core spends about 2 ms per 1280x1024 frame, two frames per batch; the
rest of the hardware latency is the CPU copy out with the tone, which BGR
lengthens by the UYVY-to-BGR conversion. With BGR output the hardware ISP
trades 1.3 ms of latency for about 0.4 cores; with UYVY it is faster and
saves more than a core.

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
BGR888 (`CV_8UC3`, `bgr8`) by default on every platform, with the RGB tone
mapping applied; on RK3576 and RK3588 it comes from the hardware ISP.
`capture_image_imu --output-yuv422` selects packed UYVY422 instead (`CV_8UC2`,
byte order `Cb,Y0,Cr,Y1`, fast-balanced ISP only). It uses BT.601 full-range
values and vertically duplicates the ISP's 4:2:0 chroma rows, with the same
digital exposure and tone. Do not decode this stream as studio/limited-range
YUV. The quality-reference ISP always outputs BGR888.

On RK3588, the four-camera sample is additionally headless by default and
restricts the complete capture process to the four Cortex-A76 cores CPU4-7
before any UVC, OpenCV, or ISP thread is created. The Cortex-A55 cores CPU0-3
are not used. The sample also disables LITTLE-core assistance and nested
per-frame ISP sharding.

Useful overrides are:

```c
# Show the local preview explicitly.
./capture_image_imu --display --isp-fast

# Select BGR888 explicitly (the default).
./capture_image_imu --output-bgr --isp-fast

# Select packed full-range UYVY422 explicitly.
./capture_image_imu --output-yuv422 --isp-fast

# Restore per-frame metadata logging (RK3588 defaults to one sample/second).
./capture_image_imu --verbose
```
