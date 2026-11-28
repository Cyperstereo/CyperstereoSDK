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
"""EuRoC folder format (a.k.a. mav0 layout) → ROS1 bag.

Writes /camN/image_raw (mono8) for every full-rate camN/
directory and /imu0 when present, in time order.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
# rosbags + PIL are optional extras: everything below is resolved LAZILY on
# first use so this module imports with no extras installed (CI wheel smoke).
# PEP 562 __getattr__ serves external imports of TYPESTORE/_Image etc.;
# _ensure_types() populates module globals for internal references.
_LAZY_NAMES = ("TYPESTORE", "_Image", "_Imu", "_Header", "_Time",
               "_Vector3", "_Quaternion", "PILImage", "Reader", "Writer")


def _ensure_types():
    if "TYPESTORE" in globals():
        return
    from PIL import Image as PILImage
    from rosbags.rosbag1 import Reader, Writer
    from rosbags.typesys import Stores, get_typestore
    ts = get_typestore(Stores.ROS1_NOETIC)
    globals().update(
        TYPESTORE=ts,
        _Image=ts.types["sensor_msgs/msg/Image"],
        _Imu=ts.types["sensor_msgs/msg/Imu"],
        _Header=ts.types["std_msgs/msg/Header"],
        _Time=ts.types["builtin_interfaces/msg/Time"],
        _Vector3=ts.types["geometry_msgs/msg/Vector3"],
        _Quaternion=ts.types["geometry_msgs/msg/Quaternion"],
        PILImage=PILImage,
        Reader=Reader,
        Writer=Writer,
    )


def _types():
    """Sibling-module door to the lazy rosbags/PIL surface: a plain function
    attribute, so importing IT never triggers the optional deps; calling it
    does. Verbatim-embeddable in the standalone recipe script."""
    from types import SimpleNamespace
    _ensure_types()
    return SimpleNamespace(**{k: globals()[k] for k in _LAZY_NAMES})


def __getattr__(name):
    if name in _LAZY_NAMES:
        _ensure_types()
        return globals()[name]
    raise AttributeError(name)


def _header(timestamp_ns: int, frame_id: str):
    _ensure_types()
    return _Header(seq=0,
                   stamp=_Time(sec=int(timestamp_ns // 1_000_000_000),
                               nanosec=int(timestamp_ns % 1_000_000_000)),
                   frame_id=frame_id)


def _csv_rows(path: Path):
    with open(path) as f:
        for row in csv.reader(f):
            if row and not row[0].startswith("#"):
                yield row


def convert_euroc_folder_to_rosbag(source_dir: str | Path,
                                   output_bag: str | Path,
                                   *, max_frames: int = 0) -> dict:
    """Convert one EuRoC-layout folder (…/mav0) into ``output_bag``.

    Returns ``{"images": n, "imu_samples": n, "cameras": [...]}``.
    """
    _ensure_types()
    source_dir, output_bag = Path(source_dir), Path(output_bag)
    if source_dir.name != "mav0" and (source_dir / "mav0").is_dir():
        source_dir = source_dir / "mav0"
    output_bag.parent.mkdir(parents=True, exist_ok=True)
    if output_bag.exists():
        output_bag.unlink()

    cameras = sorted(d.name for d in source_dir.iterdir()
                     if re.fullmatch(r"cam\d+", d.name)
                     and (d / "data.csv").is_file())
    # Monado MSD imustatic sequences ship only imu0/data.raw.csv
    imu_csv = source_dir / "imu0" / "data.csv"
    if not imu_csv.is_file():
        imu_csv = source_dir / "imu0" / "data.raw.csv"
    if not cameras and not imu_csv.is_file():
        raise FileNotFoundError(
            f"neither cam<N>/data.csv nor imu0/data[.raw].csv "
            f"under {source_dir}")

    messages: list[tuple[int, str, object]] = []
    for camera in cameras:
        count = 0
        for timestamp_ns, filename in _csv_rows(source_dir / camera / "data.csv"):
            messages.append((int(timestamp_ns), camera,
                             source_dir / camera / "data" / filename.strip()))
            count += 1
            if max_frames and count >= max_frames:
                break
    # IMU-only folders can be enormous (Monado imustatic: 174M rows, 16 GB
    # csv) — stream them straight into the writer instead of materializing;
    # the csv is already time-sorted. With cameras present the merged list
    # stays small enough to sort in memory.
    if cameras and imu_csv.is_file():
        for row in _csv_rows(imu_csv):
            gyro = list(map(float, row[1:4]))
            accel = list(map(float, row[4:7]))
            messages.append((int(row[0]), "imu0", (gyro, accel)))
    messages.sort(key=lambda m: m[0])
    if not cameras and imu_csv.is_file():
        messages = ((int(row[0]), "imu0",
                     (list(map(float, row[1:4])), list(map(float, row[4:7]))))
                    for row in _csv_rows(imu_csv))
    if max_frames and cameras:
        last_camera_ts = max(t for t, kind, _ in messages
                             if kind.startswith("cam"))
        messages = [m for m in messages if m[0] <= last_camera_ts]

    image_count = imu_count = 0
    with Writer(output_bag) as writer:
        connections = {camera: writer.add_connection(
            f"/{camera}/image_raw", _Image.__msgtype__, typestore=TYPESTORE)
            for camera in cameras}
        if imu_csv.is_file():
            connections["imu0"] = writer.add_connection(
                "/imu0", _Imu.__msgtype__, typestore=TYPESTORE)
        for timestamp_ns, kind, payload in messages:
            if kind == "imu0":
                gyro, accel = payload
                message = _Imu(
                    header=_header(timestamp_ns, "imu0"),
                    orientation=_Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                    orientation_covariance=np.array([-1.0] + [0.0] * 8),
                    angular_velocity=_Vector3(*map(float, gyro)),
                    angular_velocity_covariance=np.zeros(9),
                    linear_acceleration=_Vector3(*map(float, accel)),
                    linear_acceleration_covariance=np.zeros(9))
                writer.write(connections["imu0"], timestamp_ns,
                             TYPESTORE.serialize_ros1(message,
                                                      _Imu.__msgtype__))
                imu_count += 1
            else:
                from mvis.io.image_decode import image_file_to_gray
                array = image_file_to_gray(payload)
                if array is None:
                    raise OSError(f"cannot read image {payload}")
                height, width = array.shape
                message = _Image(header=_header(timestamp_ns, kind),
                                 height=height, width=width, encoding="mono8",
                                 is_bigendian=0, step=width,
                                 data=array.reshape(-1))
                writer.write(connections[kind], timestamp_ns,
                             TYPESTORE.serialize_ros1(message,
                                                      _Image.__msgtype__))
                image_count += 1
    return {"images": image_count, "imu_samples": imu_count,
            "cameras": cameras}
