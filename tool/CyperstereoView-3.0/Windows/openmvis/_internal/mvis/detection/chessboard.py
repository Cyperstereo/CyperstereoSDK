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
"""Chessboard calibration board detection."""

import numpy as np
import cv2

from mvis.detection.board import Board
from mvis.core.types import Observation


class Chessboard(Board):
    """Standard chessboard pattern detection using OpenCV.

    The board is defined by inner corner count (rows x cols) and square size.
    For example, a standard 10x7 chessboard has 9x6 inner corners.
    """

    # Chessboard has no notion of "tags" — every detected corner stands
    # alone. Set _num_tags = 0 so callers that branch on this attribute
    # (ablation script, report tag-count divisor) treat each corner as
    # one observation.
    _num_tags: int = 0

    def __init__(self, rows: int, cols: int, square_size: float):
        """
        Args:
            rows: Number of inner corners along rows.
            cols: Number of inner corners along columns.
            square_size: Size of each square in meters.
        """
        self._rows = rows
        self._cols = cols
        self._square_size = square_size
        # Chessboard "tag_size" / "tag_family" don't exist on this board;
        # alias them so generic helpers (detection cache key, etc.) can
        # use getattr() without choking.
        self._tag_size = float(square_size)
        self._tag_family = "chessboard"

        # Precompute 3D board points (z=0 plane)
        self._points_3d = np.zeros((rows * cols, 3), dtype=np.float64)
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                self._points_3d[idx] = [j * square_size, i * square_size, 0.0]

    def detect(self, image: np.ndarray) -> list[Observation]:
        """Detect the full inner-corner grid via cv2.findChessboardCorners + subpixel
        refinement. All-or-nothing: returns [] unless the whole grid is found
        (chessboard corners are indistinguishable, so partial grids are ambiguous).
        Point IDs are row-major grid indices; rolling-shutter rows are set."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_FAST_CHECK
        )
        ret, corners = cv2.findChessboardCorners(
            gray, (self._cols, self._rows), flags=flags
        )

        if not ret or corners is None:
            return []

        # Sub-pixel refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        observations = []
        for idx in range(len(corners)):
            px = corners[idx].flatten()
            observations.append(
                Observation(
                    point_id=idx,
                    pixel=px,
                    row=int(px[1]),
                )
            )
        return observations

    def point_3d(self, point_id: int) -> np.ndarray:
        """Board-frame 3D position of inner corner `point_id` (Board interface)."""
        return self._points_3d[point_id].copy()

    def num_points(self) -> int:
        """Total inner-corner count, rows x cols (Board interface)."""
        return self._rows * self._cols

    def all_points_3d(self) -> np.ndarray:
        """All inner corners as an (N,3) array in row-major ID order (Board interface)."""
        return self._points_3d.copy()

    def min_points_for_pose(self) -> int:
        """Minimum detections for a usable PnP pose (Board interface)."""
        return 4

    @property
    def rows(self) -> int:
        """Inner-corner rows."""
        return self._rows

    @property
    def cols(self) -> int:
        """Inner-corner columns."""
        return self._cols

    @property
    def square_size(self) -> float:
        """Square edge length (m)."""
        return self._square_size
