# SmartSens metadata compatibility test

This test covers the platform-neutral firmware marker, Bayer phase and
metadata-column layouts used by the Windows, Linux/V4L2, ROS and ROS 2 builds.
It verifies firmware 2/3, 2/4, 2/5 and 2/6, including the 13-slot IMU layout
and the HTS=452 exposure-time conversion used by firmware 2/6.

Native Linux or on-device ARM build:

```sh
cmake -S tests -B tests/build -DCMAKE_BUILD_TYPE=Release
cmake --build tests/build --parallel
ctest --test-dir tests/build --output-on-failure
```

For an ARM cross build, pass the deployment toolchain file in the usual way.
The executable must be run on the target (or under an appropriate emulator):

```sh
cmake -S tests -B tests/build-arm \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/arm-linux-toolchain.cmake
cmake --build tests/build-arm --parallel
```
