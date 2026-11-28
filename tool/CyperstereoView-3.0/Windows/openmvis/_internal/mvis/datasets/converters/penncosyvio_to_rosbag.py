# OpenMVIS: A Multi-Visual-Inertial Sensor Calibration Benchmark
# Copyright (C) 2026 Yulin Yang
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-FileCopyrightText: 2026 Yulin Yang <contact@openmvis.com>
# SPDX-License-Identifier: GPL-3.0-only
"""PennCOSYVIO sensor sequence (mp4 + timestamps [+ IMU txt]) → ROS1 bag.

Sequence directory layout (per as/af/bs/bf inside a sensor tar):
    seq_<name>.mp4            # video
    timestamps.txt            # one frame timestamp [s] per line
    accelerometer.txt         # optional: t ax ay az
    gyroscope.txt             # optional: t gx gy gz

Decoding follows the packet-exact PyAV pattern proven on earlier MCAP
video data. Some distributed videos (c3 af/bf) hold fewer coded frames than
timestamps.txt lists — the mp4 sample table claims the full count but the
trailing samples are absent, and modern ffmpeg decodes the same short count
(the 2016 devkit relied on old ffmpeg's CFR frame duplication to pad it).
Frames are contiguous from the start, so frame i pairs with timestamp i and
surplus trailing stamps are dropped (recorded in the returned stats). Fewer
timestamps than frames still raises: that mapping would be ambiguous.

PennCOSYVIO timestamps are RELATIVE seconds on a shared sync timeline and
start negative (frames before the sync zero). Bags cannot hold negative
stamps, so a fixed EPOCH_OFFSET_S is added uniformly to camera and IMU
times — identical for every sensor, so cross-sensor sync is preserved —
and recorded in the returned stats / provenance sidecar.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .euroc_folder_to_rosbag import _header, _types


EPOCH_OFFSET_S = 1000.0


def _load_imu(sequence_dir: Path):
    accel_file = sequence_dir / "accelerometer.txt"
    gyro_file = sequence_dir / "gyroscope.txt"
    if not (accel_file.is_file() and gyro_file.is_file()):
        return None
    accel = np.loadtxt(accel_file)
    gyro = np.loadtxt(gyro_file)
    return gyro, accel


def convert_penncosyvio_sequence_to_rosbag(sequence_dir: str | Path,
                                           output_bag: str | Path, *,
                                           camera_topic: str = "/cam0/image_raw",
                                           imu_topic: str = "/imu0") -> dict:
    try:
        import av  # deferred: PyAV only needed for this dataset family (the `mcap` extra)
    except ImportError as e:
        raise ImportError("PyAV is needed for this converter: pip install av") from e
    _T = _types()  # lazy rosbags/PIL surface (CI wheel smoke)

    sequence_dir, output_bag = Path(sequence_dir), Path(output_bag)
    (video_path,) = sorted(sequence_dir.glob("*.mp4"))
    frame_times_s = np.loadtxt(sequence_dir / "timestamps.txt")
    imu = _load_imu(sequence_dir)
    output_bag.parent.mkdir(parents=True, exist_ok=True)
    if output_bag.exists():
        output_bag.unlink()

    image_count = imu_count = 0
    with _T.Writer(output_bag) as writer:
        camera_connection = writer.add_connection(
            camera_topic, _T._Image.__msgtype__, typestore=_T.TYPESTORE)
        imu_connection = writer.add_connection(
            imu_topic, _T._Imu.__msgtype__, typestore=_T.TYPESTORE) if imu else None

        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            for index, frame in enumerate(container.decode(stream)):
                if index >= len(frame_times_s):
                    raise ValueError(
                        f"{video_path.name}: more decoded frames than the "
                        f"{len(frame_times_s)} timestamps — frame/stamp "
                        f"mapping would be ambiguous")
                timestamp_ns = int(round((float(frame_times_s[index])
                                          + EPOCH_OFFSET_S) * 1e9))
                array = frame.to_ndarray(format="gray")
                height, width = array.shape
                message = _T._Image(header=_header(timestamp_ns, "cam0"),
                                 height=height, width=width,
                                 encoding="mono8", is_bigendian=0, step=width,
                                 data=array.reshape(-1))
                writer.write(camera_connection, timestamp_ns,
                             _T.TYPESTORE.serialize_ros1(message,
                                                      _T._Image.__msgtype__))
                image_count += 1
        unmatched_trailing_stamps = len(frame_times_s) - image_count

        if imu is not None:
            gyro, accel = imu
            for g_row, a_row in zip(gyro, accel):
                timestamp_ns = int(round((float(g_row[0])
                                          + EPOCH_OFFSET_S) * 1e9))
                message = _T._Imu(
                    header=_header(timestamp_ns, "imu0"),
                    orientation=_T._Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                    orientation_covariance=np.array([-1.0] + [0.0] * 8),
                    angular_velocity=_T._Vector3(*map(float, g_row[1:4])),
                    angular_velocity_covariance=np.zeros(9),
                    linear_acceleration=_T._Vector3(*map(float, a_row[1:4])),
                    linear_acceleration_covariance=np.zeros(9))
                writer.write(imu_connection, timestamp_ns,
                             _T.TYPESTORE.serialize_ros1(message,
                                                      _T._Imu.__msgtype__))
                imu_count += 1
    return {"images": image_count, "imu_samples": imu_count,
            "timestamp_offset_applied_s": EPOCH_OFFSET_S,
            "unmatched_trailing_stamps": unmatched_trailing_stamps}
