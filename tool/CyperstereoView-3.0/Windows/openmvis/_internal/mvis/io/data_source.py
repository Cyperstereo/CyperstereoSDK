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
"""Abstract data source interface."""

from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np


class DataSource(ABC):
    """Abstract base class for calibration image sources."""

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[np.ndarray, float, int]]:
        """Iterate over frames.

        Yields:
            (image, timestamp, camera_id) tuples.
        """

    @abstractmethod
    def num_cameras(self) -> int:
        """Return the number of cameras in this data source."""


_ROS1_TYPES = {"rosbag_topic", "ros1bag"}


def make_sources(cameras) -> list[tuple[DataSource, list[int]]]:
    """The camera-only reader factory: one ``DataSource`` per bag (all of that bag's
    camera topics, read in bag order) and one per image folder / EuRoC / timestamped
    camera, each with the config camera ids of its local camera indices. No reader-side
    subsampling (``target_hz=0``): rate selection runs after detection.

    ``cameras`` are ``CameraConfig``s; source types ``images``, ``timestamped_images``
    (``source_path`` = ``image_dir,timestamps.csv``), ``euroc``, ``rosbag_topic`` /
    ``ros1bag`` and ``ros2bag``. Raises ValueError for an unknown type or a bag camera
    without a topic."""
    groups: dict[tuple[str, str], list] = {}
    for cam in cameras:
        kind = "ros1bag" if cam.source_type in _ROS1_TYPES else cam.source_type
        if kind not in ("images", "timestamped_images", "euroc", "ros1bag", "ros2bag"):
            raise ValueError(f"source type not supported: {cam.source_type!r}")
        if kind in ("ros1bag", "ros2bag") and not cam.topic:
            raise ValueError(f"camera {cam.camera_id}: a {kind} source needs a topic")
        key = (kind, cam.source_path) if kind.endswith("bag") else (kind, id(cam))
        groups.setdefault(key, []).append(cam)

    out = []
    for (kind, _), cams in groups.items():
        c = cams[0]
        if kind == "ros1bag":
            from mvis.io.ros1_bag_source import Ros1BagSource
            src = Ros1BagSource(c.source_path, [x.topic for x in cams])
        elif kind == "ros2bag":
            from mvis.io.ros2_bag_source import Ros2BagSource
            src = Ros2BagSource(c.source_path, [x.topic for x in cams])
        elif kind == "images":
            from mvis.io.image_source import ImageSource
            src = ImageSource(c.source_path)
        elif kind == "euroc":
            from mvis.io.euroc_image_source import EurocImageSource
            src = EurocImageSource(c.source_path, camera_id=0)
        else:
            from mvis.io.timestamped_image_source import TimestampedImageSource
            image_dir, stamps = c.source_path.split(",")
            src = TimestampedImageSource(image_dir, stamps, camera_id=0)
        out.append((src, [x.camera_id for x in cams]))
    return out


def stream_cameras(cameras):
    """Yield ``(camera_id, image, timestamp)`` for every frame of ``make_sources(cameras)``,
    source after source, one image at a time."""
    for src, cam_ids in make_sources(cameras):
        for image, timestamp, i in src:
            yield cam_ids[i], image, timestamp
