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
"""v3 — per-camera reprojection stats on result_camera and overall_rms on
run (the calib-YAML export view needs them). Pure ADD COLUMN, no
rebuild; all nullable so v2 rows stay valid."""

VERSION = 3

_COLS = [
    ("result_camera", "rms", "REAL"), ("result_camera", "mean_err", "REAL"),
    ("result_camera", "median_err", "REAL"), ("result_camera", "p90", "REAL"),
    ("result_camera", "p95", "REAL"), ("result_camera", "n_obs", "INTEGER"),
    ("run", "overall_rms", "REAL"),
]


def up(conn) -> None:
    for table, col, typ in _COLS:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
