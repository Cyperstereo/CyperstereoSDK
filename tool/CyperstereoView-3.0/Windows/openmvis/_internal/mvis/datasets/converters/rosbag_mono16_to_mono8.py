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
"""ROS1 bag mono16→mono8 re-encode + header-clock re-stamp.

Fixes datasets (e.g. TUM-VI raw) whose bag-record time differs from the
header clock, and makes mono16 images detectable by the AprilGrid detector.
Pixels follow ``io.image_decode.mono16_to_mono8``, so the converted bag reads exactly
like the raw bag does in every MVIS reader.
"""
from __future__ import annotations

from pathlib import Path

from mvis.io.image_decode import ros_image_to_gray

from .euroc_folder_to_rosbag import _types


def convert_rosbag_mono16_to_mono8(
        input_bag: str | Path, output_bag: str | Path, *,
        image_topics=("/cam0/image_raw", "/cam1/image_raw"),
        keep_topics=("/cam0/image_raw", "/cam1/image_raw", "/imu0")) -> dict:
    _T = _types()  # lazy rosbags surface (CI wheel smoke)
    input_bag, output_bag = Path(input_bag), Path(output_bag)
    output_bag.parent.mkdir(parents=True, exist_ok=True)
    if output_bag.exists():
        output_bag.unlink()
    image_topics, keep_topics = set(image_topics), set(keep_topics)
    converted = passthrough = 0
    with _T.Reader(input_bag) as reader, _T.Writer(output_bag) as writer:
        connections = {}
        for connection in reader.connections:
            if connection.topic in keep_topics:
                connections[connection.id] = writer.add_connection(
                    connection.topic, connection.msgtype, typestore=_T.TYPESTORE)
        for connection, record_time, raw in reader.messages():
            if connection.topic not in keep_topics:
                continue
            out_connection = connections[connection.id]
            message = _T.TYPESTORE.deserialize_ros1(raw, connection.msgtype)
            header = getattr(message, "header", None)
            write_time = (int(header.stamp.sec) * 1_000_000_000
                          + int(header.stamp.nanosec)) \
                if header is not None else record_time
            if connection.topic in image_topics and \
                    getattr(message, "encoding", "") in ("mono16", "16UC1"):
                as_mono8 = ros_image_to_gray(message, connection.topic)
                message.encoding = "mono8"
                message.is_bigendian = 0
                message.step = message.width
                message.data = as_mono8.reshape(-1)
                writer.write(out_connection, write_time,
                             _T.TYPESTORE.serialize_ros1(message,
                                                      connection.msgtype))
                converted += 1
            else:
                writer.write(out_connection, write_time, raw)
                passthrough += 1
    return {"converted": converted, "passthrough": passthrough}
