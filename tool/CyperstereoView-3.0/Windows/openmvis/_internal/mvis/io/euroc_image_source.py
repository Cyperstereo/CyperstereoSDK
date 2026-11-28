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
"""EuRoC / ASL-format image source.

Reads image collections stored in the EuRoC convention::

    <cam_dir>/
        data.csv           # "#timestamp [ns],filename" header + rows
        data/
            <stamp>.png
            ...

Timestamps in the CSV are nanoseconds; they are converted to seconds
when yielded so downstream code (SLERP, trajectory alignment) can
stay unit-agnostic.

This is a sibling of ``TimestampedImageSource`` — we don't reuse that
class because its CSV order is ``filename,timestamp`` with no header,
which is NOT the EuRoC convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from mvis.io.data_source import DataSource
from mvis.io.image_decode import image_file_to_gray


class EurocImageSource(DataSource):
    """Iterate (image, timestamp_seconds, camera_id) from an EuRoC directory.

    Expects::

        <cam_dir>/data.csv      header "#timestamp [ns],filename"
        <cam_dir>/data/<file>   image file referenced in each row
    """

    def __init__(self, cam_dir: str, camera_id: int = 0):
        self._cam_dir = Path(cam_dir)
        self._camera_id = camera_id
        self._img_dir = self._cam_dir / "data"
        csv_path = self._cam_dir / "data.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"EuRoC data.csv not found: {csv_path}")
        if not self._img_dir.is_dir():
            raise FileNotFoundError(f"EuRoC data/ dir not found: {self._img_dir}")

        entries: list[tuple[int, Path]] = []
        with open(csv_path) as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2:
                    continue
                try:
                    ts_ns = int(parts[0])
                except ValueError:
                    continue
                fp = self._img_dir / parts[1]
                entries.append((ts_ns, fp))

        # EuRoC CSVs are usually sorted but we don't depend on it.
        entries.sort(key=lambda e: e[0])
        self._entries = entries

    def num_cameras(self) -> int:
        """Number of cameras this source provides."""
        return 1

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[tuple[np.ndarray, float, int]]:
        for ts_ns, fp in self._entries:
            img = image_file_to_gray(fp)   # uint8 grey, the rule every reader shares
            if img is None:
                continue
            yield img, float(ts_ns) * 1e-9, self._camera_id
