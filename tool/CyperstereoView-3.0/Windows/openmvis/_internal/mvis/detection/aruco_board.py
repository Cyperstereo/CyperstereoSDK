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
"""ArUco calibration board detection with adaptive upscaling."""

import logging

import numpy as np

from mvis.detection.board import Board, LazyDetector, scaled_to_original
from mvis.core.types import Observation

logger = logging.getLogger(__name__)

MIN_TAG_PIXEL_SIZE = 20.0
MIN_DETECTION_RATE = 0.85


# Mapping from string names to OpenCV ArUco dictionary IDs
_ARUCO_DICT_MAP = {
    "DICT_4X4_50": 0,
    "DICT_4X4_100": 1,
    "DICT_4X4_250": 2,
    "DICT_4X4_1000": 3,
    "DICT_5X5_50": 4,
    "DICT_5X5_100": 5,
    "DICT_5X5_250": 6,
    "DICT_5X5_1000": 7,
    "DICT_6X6_50": 8,
    "DICT_6X6_100": 9,
    "DICT_6X6_250": 10,
    "DICT_6X6_1000": 11,
    "DICT_7X7_50": 12,
    "DICT_7X7_100": 13,
    "DICT_7X7_250": 14,
    "DICT_7X7_1000": 15,
}


def aruco_detector(aruco_dict: str, refine_subpixel: bool = False):
    """A cv2 ``ArucoDetector`` for the named dictionary, with OpenCV's subpixel corner
    refinement when `refine_subpixel` (cv2 import deferred: an optional dependency)."""
    import cv2
    aruco = cv2.aruco
    dict_id = _ARUCO_DICT_MAP.get(aruco_dict)
    if dict_id is None:
        raise ValueError(f"Unknown ArUco dictionary: {aruco_dict}")
    params = aruco.DetectorParameters()
    if refine_subpixel:
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    return aruco.ArucoDetector(aruco.getPredefinedDictionary(dict_id), params)


class ArucoBoard(LazyDetector, Board):
    """ArUco marker board detection using OpenCV.

    Requires opencv-contrib-python: pip install opencv-contrib-python
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        marker_size: float,
        marker_separation: float,
        aruco_dict: str = "DICT_6X6_250",
        min_tag_px: float = MIN_TAG_PIXEL_SIZE,
        min_detection_rate: float = MIN_DETECTION_RATE,
    ):
        """
        Args:
            rows: Number of marker rows.
            cols: Number of marker columns.
            marker_size: Marker size in meters.
            marker_separation: Separation between markers in meters.
            aruco_dict: ArUco dictionary name.
        """
        self._rows = rows
        self._cols = cols
        self._marker_size = marker_size
        self._marker_separation = marker_separation
        self._aruco_dict_name = aruco_dict
        self._num_markers = rows * cols
        self._min_tag_px = min_tag_px
        self._min_detection_rate = min_detection_rate
        # in the board fingerprint, so detection caches written before the F6 upscale fix
        # (corners mapped back with x / scale) miss and re-detect
        self._corner_map = "pixel_centre"

        # Precompute 3D points: 4 corners per marker (z=0 plane)
        step = marker_size + marker_separation
        half = marker_size / 2.0
        corner_offsets = np.array([
            [-half, -half, 0],
            [half, -half, 0],
            [half, half, 0],
            [-half, half, 0],
        ])

        self._points_3d = {}
        for r in range(rows):
            for c in range(cols):
                marker_idx = r * cols + c
                center = np.array([c * step, r * step, 0.0])
                for corner_idx in range(4):
                    point_id = marker_idx * 4 + corner_idx
                    self._points_3d[point_id] = center + corner_offsets[corner_idx]

    def _make_detector(self):
        """The cv2 detector, built once on first use (default parameters, no refinement)."""
        return aruco_detector(self._aruco_dict_name)

    def _detect_raw(self, gray: np.ndarray):
        """Run ArUco detection on a grayscale image. Returns (corners_list, ids)."""
        corners_list, ids, _ = self._get_detector().detectMarkers(gray)
        return corners_list, ids

    def detect(self, image: np.ndarray) -> list[Observation]:
        """Detect ArUco markers with adaptive upscaling.

        Strategy: detect at original resolution first. If median marker size
        < min_tag_px or detection rate < min_detection_rate, upscale and retry.
        """
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "ArUco detection requires 'opencv-contrib-python'. "
                "Install with: pip install opencv-contrib-python"
            )

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # First pass
        corners_list, ids = self._detect_raw(gray)
        scale = 1.0

        n_detected = 0 if ids is None else len(ids)
        detection_rate = n_detected / self._num_markers if self._num_markers > 0 else 1.0

        # Check quality
        need_upscale = False
        if n_detected > 0:
            sizes = []
            for c in corners_list:
                pts = c.reshape(4, 2)
                max_side = max(np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4))
                sizes.append(max_side)
            median_size = float(np.median(sizes))

            if median_size < self._min_tag_px:
                logger.warning(
                    f"ArUco detection: median marker size {median_size:.1f}px < "
                    f"{self._min_tag_px}px. Upscaling."
                )
                need_upscale = True
                scale = self._min_tag_px / median_size * 1.5

        if detection_rate < self._min_detection_rate:
            if n_detected > 0:
                logger.warning(
                    f"ArUco detection: {n_detected}/{self._num_markers} markers "
                    f"({detection_rate:.0%}) < {self._min_detection_rate:.0%}. Upscaling."
                )
            need_upscale = True
            if scale <= 1.0:
                scale = 2.0

        # Second pass with upscaling. The native corners are kept unless the upscaled
        # image finds more markers: upscaled corners carry the resize blur.
        if need_upscale:
            scale = max(scale, 1.5)
            scale = min(scale, 4.0)
            gray_up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            corners_up, ids_up = self._detect_raw(gray_up)

            n_after = 0 if ids_up is None else len(ids_up)
            if n_after > n_detected:
                logger.info(f"ArUco upscale ({scale:.1f}x): {n_detected} -> {n_after} markers")
                corners_list, ids = corners_up, ids_up
            else:
                scale = 1.0

        if ids is None:
            return []

        observations = []
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id < 0 or marker_id >= self._num_markers:
                continue

            corners = corners_list[i].reshape(4, 2)
            if scale != 1.0:
                corners = scaled_to_original(corners, scale)

            for corner_idx in range(4):
                point_id = marker_id * 4 + corner_idx
                px = corners[corner_idx]
                observations.append(
                    Observation(
                        point_id=point_id,
                        pixel=px,
                        row=int(px[1]),
                    )
                )

        return observations

    def point_3d(self, point_id: int) -> np.ndarray:
        """Board-frame 3D position of corner `point_id` (Board interface)."""
        return self._points_3d[point_id].copy()

    def num_points(self) -> int:
        """Total corner count: 4 per marker (Board interface)."""
        return self._num_markers * 4

    def min_points_for_pose(self) -> int:
        """Minimum detections for a usable PnP pose — one full marker (Board interface)."""
        return 4
