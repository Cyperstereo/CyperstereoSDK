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
"""Abstract calibration board interface."""

from abc import ABC, abstractmethod

import numpy as np

from mvis.core.types import Observation


def scaled_to_original(corners: np.ndarray, scale: float,
                       offset: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Map pixel coords detected in a ``cv2.resize(..., fx=scale, fy=scale)``
    image back to the source image (plus a crop ``offset``).

    cv2.resize aligns pixel CENTRES: x_dst = (x_src + 0.5) * scale - 0.5, so
    the inverse is (x_dst + 0.5) / scale - 0.5. Plain ``x_dst / scale``
    overshoots by +(0.5 - 0.5 / scale): +0.25 px at 2x, +0.375 px at 4x."""
    c = np.asarray(corners, dtype=np.float64)
    return (c + 0.5) / scale - 0.5 + np.asarray(offset, dtype=np.float64)


class LazyDetector:
    """Mixin for detectors whose backend object is built on first use (so an optional
    dependency is imported only when detection actually runs): subclasses implement
    ``_make_detector()``; ``_get_detector()`` builds it once and caches it."""

    _detector = None

    def _make_detector(self):
        raise NotImplementedError

    def _get_detector(self):
        if self._detector is None:
            self._detector = self._make_detector()
        return self._detector

    def __getstate__(self):
        # copies and pickles rebuild the backend: a cv2 detector cannot be pickled
        state = dict(vars(self))
        state.pop("_detector", None)
        return state


class BoardGeometry(ABC):
    """The TARGET half of the old Board contract: pure
    geometry, no algorithm. This is what the store persists (db.boards) and
    what bridge.board_from_geometry consumes. New code should type against
    THIS; `Board` below keeps the legacy combined contract while call sites
    migrate to the FeatureDetector hierarchy in mvis.detection."""

    @abstractmethod
    def point_3d(self, point_id: int) -> np.ndarray:
        """Get the 3D position of a board point in the board frame.

        Args:
            point_id: Board point ID.

        Returns:
            3D position [x, y, z] in meters (z=0 for planar boards).
        """

    @abstractmethod
    def num_points(self) -> int:
        """Total number of points on the board."""

    def all_points_3d(self) -> np.ndarray:
        """Return Nx3 array of all board points in the board frame, indexed by point ID
        (from `point_3d`; a board with its points in one array may return a copy instead)."""
        n = self.num_points()
        pts = np.zeros((n, 3), dtype=np.float64)
        for pid in range(n):
            pts[pid] = self.point_3d(pid)
        return pts

    @abstractmethod
    def min_points_for_pose(self) -> int:
        """Minimum number of detected points needed for PnP pose estimation."""


class Board(BoardGeometry):
    """Legacy combined contract: geometry + detect(). detect() belongs to the
    detectors (mvis.detection) — it stays abstract here only until the
    remaining direct call sites (frame_detector, a few scripts) migrate."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Observation]:
        """Detect board features in an image.

        Args:
            image: Input image (BGR or grayscale).

        Returns:
            List of Observation objects. Empty list if detection fails.
        """

