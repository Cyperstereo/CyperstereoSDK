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
"""Timestamped image data source."""

import csv
from pathlib import Path
from typing import Iterator

import numpy as np

from mvis.io.data_source import DataSource
from mvis.io.image_decode import image_file_to_gray


class TimestampedImageSource(DataSource):
    """Load images with associated timestamps from a CSV file.

    CSV format: filename,timestamp
    """

    def __init__(self, image_dir: str, timestamp_file: str, camera_id: int = 0):
        """
        Args:
            image_dir: Directory containing images.
            timestamp_file: Path to CSV file with (filename, timestamp) rows.
            camera_id: Camera ID for this source.
        """
        self._image_dir = Path(image_dir)
        self._camera_id = camera_id
        self._entries: list[tuple[str, float]] = []

        with open(timestamp_file) as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    filename = row[0].strip()
                    timestamp = float(row[1].strip())
                    self._entries.append((filename, timestamp))

    def __iter__(self) -> Iterator[tuple[np.ndarray, float, int]]:
        for filename, timestamp in self._entries:
            filepath = self._image_dir / filename
            image = image_file_to_gray(str(filepath))
            if image is None:
                continue
            yield image, timestamp, self._camera_id

    def num_cameras(self) -> int:
        """Number of cameras this source provides."""
        return 1
