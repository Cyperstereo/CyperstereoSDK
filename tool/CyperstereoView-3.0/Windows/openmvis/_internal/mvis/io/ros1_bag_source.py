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
"""ROS1 bag data source.

Uses the ``rosbags`` library (no ROS install required), matching the
camera-only reader of `io.data_source.make_sources`. Frames (raw Image in any encoding
io.image_decode supports, or CompressedImage PNG / JPEG) come out as uint8
grey through io.image_decode; every frame (rate selection runs after detection).
"""

from typing import Iterator

import numpy as np

from mvis.io.data_source import DataSource
from mvis.io.image_decode import ros_image_to_gray
from mvis.io.ros_stamp import header_stamp_s  # noqa: F401  (its home is io.ros_stamp; old import path)


class Ros1BagSource(DataSource):
    """Load images from a ROS1 bag file using the rosbags library."""

    def __init__(self, bag_path: str, topics: list[str]):
        """
        Args:
            bag_path: Path to .bag file.
            topics: List of image topic names (one per camera).
        """
        self._bag_path = bag_path
        self._topics = topics
        self._topic_to_cam = {t: i for i, t in enumerate(topics)}

    def _decode(self, msg, topic: str = "", warned: set | None = None) -> np.ndarray:
        return ros_image_to_gray(msg, topic, warned)

    def __iter__(self) -> Iterator[tuple[np.ndarray, float, int]]:
        from rosbags.rosbag1 import Reader
        from rosbags.typesys import Stores, get_typestore

        typestore = get_typestore(Stores.ROS1_NOETIC)
        topics_set = set(self._topics)

        jpeg_warned: set[str] = set()
        with Reader(self._bag_path) as reader:
            for conn, ts, rawdata in reader.messages():
                if conn.topic not in topics_set:
                    continue
                cam_id = self._topic_to_cam[conn.topic]
                msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
                yield (self._decode(msg, conn.topic, jpeg_warned),
                       header_stamp_s(rawdata, ts), cam_id)

    def num_cameras(self) -> int:
        """Number of cameras this source provides."""
        return len(self._topics)
