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
"""Image folder → ROS1 bag with deterministic synthetic timestamps.

For still-image calibration sets (PennCOSYVIO intrinsic stills, Urban
fisheye) that carry no capture times. Stamps are index / frame_rate_hz from
``start_time_ns`` (default epoch 0), so re-runs are byte-identical and the
provenance sidecar can state ``timestamps: synthetic``. Usable for camera-only
intrinsics; never for td/cam-IMU work.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mvis.io.image_decode import image_file_to_gray

from .euroc_folder_to_rosbag import _header, _types

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm"}


def convert_image_folder_to_rosbag(image_dir: str | Path,
                                   output_bag: str | Path, *,
                                   topic: str = "/cam0/image_raw",
                                   frame_rate_hz: float = 1.0,
                                   start_time_ns: int = 0) -> dict:
    _T = _types()  # lazy rosbags/PIL surface (CI wheel smoke)
    image_dir, output_bag = Path(image_dir), Path(output_bag)
    images = sorted(p for p in image_dir.iterdir()
                    if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"no images under {image_dir}")
    output_bag.parent.mkdir(parents=True, exist_ok=True)
    if output_bag.exists():
        output_bag.unlink()
    period_ns = int(round(1e9 / frame_rate_hz))
    with _T.Writer(output_bag) as writer:
        connection = writer.add_connection(topic, _T._Image.__msgtype__,
                                           typestore=_T.TYPESTORE)
        index = 0
        skipped = []
        for path in images:
            try:
                array = image_file_to_gray(path)
                if array is None:
                    raise OSError(path)
            except OSError:
                skipped.append(path.name)   # truncated/corrupt image
                continue
            timestamp_ns = start_time_ns + index * period_ns
            height, width = array.shape
            message = _T._Image(header=_header(timestamp_ns, "cam0"),
                             height=height, width=width, encoding="mono8",
                             is_bigendian=0, step=width,
                             data=array.reshape(-1))
            writer.write(connection, timestamp_ns,
                         _T.TYPESTORE.serialize_ros1(message,
                                                  _T._Image.__msgtype__))
            index += 1
    return {"images": index, "skipped_unreadable": skipped, "topic": topic,
            "timestamps": "synthetic", "frame_rate_hz": frame_rate_hz}
