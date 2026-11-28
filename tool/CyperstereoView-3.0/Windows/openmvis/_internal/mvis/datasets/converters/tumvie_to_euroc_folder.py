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
"""TUM-VIE vi_data folder → EuRoC folder format (µs → ns, exact).

Run by ``mvis datasets convert`` for the TUM-VIE sequences.
vi_data layout: left_images/00000.jpg + image_timestamps_left.txt (µs),
right_images/..., imu_data.txt (t µs, gyro rad/s, accel m/s², temperature).
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np


def _convert_camera(source: Path, destination: Path, timestamps_file: str) -> int:
    timestamps_us = np.atleast_1d(np.loadtxt(source / timestamps_file))
    images = sorted(p for p in source.iterdir() if p.suffix == ".jpg")
    assert len(images) == len(timestamps_us), \
        f"{source}: {len(images)} images vs {len(timestamps_us)} timestamps"
    (destination / "data").mkdir(parents=True, exist_ok=True)
    with open(destination / "data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["#timestamp [ns]", "filename"])
        for image, timestamp_us in zip(images, timestamps_us):
            timestamp_ns = int(round(timestamp_us * 1e3))
            name = f"{timestamp_ns}.jpg"
            target = destination / "data" / name
            if not target.exists():
                shutil.copy2(image, target)
            writer.writerow([timestamp_ns, name])
    return len(images)


def _convert_imu(source: Path, destination: Path) -> int:
    data = np.atleast_2d(np.loadtxt(source / "imu_data.txt"))  # t_us gx gy gz ax ay az temp
    destination.mkdir(parents=True, exist_ok=True)
    with open(destination / "data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["#timestamp [ns]",
                         "w_x", "w_y", "w_z", "a_x", "a_y", "a_z"])
        for row in data:
            writer.writerow([int(round(row[0] * 1e3))]
                            + [f"{value:.9f}" for value in row[1:7]])
    return len(data)


def convert_tumvie_vidata_to_euroc_folder(sequence_dir: str | Path) -> Path:
    """Build ``<sequence_dir>/mav0`` from a TUM-VIE vi_data folder."""
    sequence_dir = Path(sequence_dir)
    euroc_dir = sequence_dir / "mav0"
    left = _convert_camera(sequence_dir / "left_images", euroc_dir / "cam0",
                           "image_timestamps_left.txt")
    right = _convert_camera(sequence_dir / "right_images", euroc_dir / "cam1",
                            "image_timestamps_right.txt")
    imu = _convert_imu(sequence_dir, euroc_dir / "imu0")
    print(f"{sequence_dir.name}: cam0 {left} + cam1 {right} images, "
          f"imu {imu} samples -> {euroc_dir}")
    return euroc_dir
