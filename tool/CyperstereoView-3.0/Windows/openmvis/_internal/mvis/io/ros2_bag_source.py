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
"""ROS2 bag data source.

Uses the ``rosbags`` library (no ROS install required), like the ROS1 source: the bag folder
with its ``metadata.yaml``, or a single ``.db3`` / ``.mcap`` file. Frames are decoded by
io.image_decode from the message bytes; cv_bridge is not needed."""

from typing import Iterator

import numpy as np

from mvis.io.data_source import DataSource
from mvis.io.image_decode import ros_image_to_gray
from mvis.io.ros_stamp import header_stamp_s
from mvis.io.rosbag_reader import open_bag


class Ros2BagSource(DataSource):
    """Load images from a ROS2 bag using the rosbags library."""

    def __init__(self, bag_path: str, topics: list[str]):
        """
        Args:
            bag_path: Path to the ROS2 bag directory (or its .db3 / .mcap file).
            topics: List of image topic names (one per camera).
        """
        self._bag_path = bag_path
        self._topics = topics
        self._topic_to_cam = {t: i for i, t in enumerate(topics)}

    def __iter__(self) -> Iterator[tuple[np.ndarray, float, int]]:
        reader, deserialize = open_bag(self._bag_path)
        jpeg_warned: set[str] = set()
        with reader:
            cons = [c for c in reader.connections if c.topic in self._topic_to_cam]
            for conn, ts, raw in reader.messages(connections=cons):
                msg = deserialize(raw, conn.msgtype)
                # header.stamp, as in the ROS1 path; record time only if it was never set
                yield (ros_image_to_gray(msg, conn.topic, jpeg_warned),
                       header_stamp_s(raw, ts), self._topic_to_cam[conn.topic])

    def num_cameras(self) -> int:
        """Number of cameras this source provides."""
        return len(self._topics)
