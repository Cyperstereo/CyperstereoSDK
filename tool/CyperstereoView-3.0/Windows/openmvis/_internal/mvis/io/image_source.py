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
"""Image directory data source."""

import os
from pathlib import Path
from typing import Iterator

import numpy as np

from mvis.io.data_source import DataSource
from mvis.io.image_decode import image_file_to_gray

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


class ImageSource(DataSource):
    """Load images from a directory.

    Images are sorted by filename and assigned sequential timestamps.
    For multi-camera setups, provide multiple directories.
    """

    def __init__(self, paths: str | list[str]):
        """
        Args:
            paths: Single directory path or list of paths (one per camera).
        """
        if isinstance(paths, str):
            paths = [paths]
        self._paths = paths
        self._image_lists: list[list[str]] = []

        for path in paths:
            p = Path(path)
            if not p.is_dir():
                raise FileNotFoundError(f"Image directory not found: {path}")
            files = sorted(
                f
                for f in p.iterdir()
                if f.suffix.lower() in _IMAGE_EXTENSIONS
            )
            self._image_lists.append([str(f) for f in files])

    def __iter__(self) -> Iterator[tuple[np.ndarray, float, int]]:
        for cam_id, image_files in enumerate(self._image_lists):
            for idx, filepath in enumerate(image_files):
                image = image_file_to_gray(filepath)
                if image is None:
                    continue
                timestamp = float(idx)
                yield image, timestamp, cam_id

    def __len__(self) -> int:
        """Total number of image files across all cameras."""
        return sum(len(fl) for fl in self._image_lists)

    def num_cameras(self) -> int:
        """Number of cameras this source provides."""
        return len(self._paths)
