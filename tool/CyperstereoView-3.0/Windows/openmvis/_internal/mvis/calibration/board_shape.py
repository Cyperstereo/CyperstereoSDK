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
"""Board-shape helpers shared by the camera-only and the cam-IMU solvers."""

from __future__ import annotations

import numpy as np


def board_height_gauge_weights(pts) -> np.ndarray:
    """(N, 3) weights of the board-height gauge rows.

    Column 0 averages the heights (1/N), columns 1 and 2 take their first moments
    about the board x and y axes, normalised by the second moments so each row is in
    metres. Pinning W^T h to zero leaves the mean plane and tilt to the board pose and
    only the departure from that plane to the heights. Used by
    `optimizer._board_height_gauge_factor` (camera-only, heights in the landmarks) and
    `pipeline_cam_imu._board_height_factors` (cam-IMU, heights key H0).
    """
    pts = np.asarray(pts, float)
    x = pts[:, 0] - pts[:, 0].mean()
    y = pts[:, 1] - pts[:, 1].mean()
    N = len(pts)
    return np.column_stack([np.full(N, 1.0 / N), x / np.sum(x * x), y / np.sum(y * y)])
