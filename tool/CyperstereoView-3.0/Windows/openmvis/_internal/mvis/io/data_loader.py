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
"""Unified data loader for all calibration image sources.

Provides a single entry point for streaming calibration images from any
supported source type. Subsampling happens after detection
(`DetectionPipeline.run_streaming`).

Usage::

    from mvis.io.data_loader import DataLoader

    loader = DataLoader(config)
    for cam_id, image, timestamp in loader.stream():
        ...
"""

import logging
from pathlib import Path

from mvis.core.config import CalibrationConfig

logger = logging.getLogger("mvis.io.data_loader")

_BAG_TYPES = {"rosbag_topic", "ros1bag", "ros2bag"}


class DataLoader:
    """Load calibration images from any supported source type.

    Source types: those of ``io.data_source.make_sources`` (images, timestamped_images,
    euroc, rosbag_topic / ros1bag, ros2bag); cameras may mix types and bags.
    """

    def __init__(self, config: CalibrationConfig):
        self._config = config
        self._source_types = {
            c.source_type for c in config.cameras if c.source_type
        }

    @property
    def is_lazy(self) -> bool:
        """True for an image-directory source (cache key = its parent dir)."""
        return "images" in self._source_types

    @property
    def cache_key(self) -> str:
        """Detection-cache key (``io.capture_key.capture_key``): the source plus the
        camera topics (bags) or camera dirs (images, EuRoC) in camera order, so two
        configs that read different cameras of one source never share a cache."""
        from mvis.io.capture_key import capture_key
        cams = sorted(self._config.cameras, key=lambda c: c.camera_id)
        if self.is_lazy or "euroc" in self._source_types:
            return capture_key(Path(cams[0].source_path).parent,
                               [c.source_path for c in cams])
        path = cams[0].source_path  # bag path
        # cameras over several bags: name every bag (one bag keeps the old key)
        paths = list(dict.fromkeys(c.source_path for c in cams))
        sources = paths if len(paths) > 1 else None
        if self._source_types & _BAG_TYPES:
            # Detections cached from record times carry the wrong stamps: keep them out.
            return capture_key(path, [c.topic for c in cams], stamp="header",
                               sources=sources)
        return capture_key(path, [c.topic for c in cams], sources=sources)

    def cam_names(self) -> dict[int, str]:
        """Return {cam_id: display_name} from config."""
        return {
            c.camera_id: (c.name or f"cam{c.camera_id}")
            for c in self._config.cameras
        }

    def stream(self):
        """Yield ``(camera_id, image, timestamp)`` one frame at a time, from
        ``io.data_source.make_sources`` (one reader per bag, one per other camera).

        No subsampling — yields every frame from the source.  Memory
        usage is bounded to one image at a time because the caller
        controls when each frame is discarded.
        """
        from mvis.io.data_source import stream_cameras
        yield from stream_cameras(self._config.cameras)
