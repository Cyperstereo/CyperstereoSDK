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
"""ArUco tags as stable point features (geometry-free front-end).

Unlike :class:`ArucoBoard`, this does NOT assume a single rigid grid layout.
It emits every detected marker's 4 corners as :class:`Observation`s keyed by a
stable, globally-unique ``point_id = marker_id*4 + corner`` — with **no
``num_markers`` ceiling** and **no board 3D geometry**.

Use it when ArUco tags are scattered across several boards at unknown relative
poses and serve only as uniquely-ID'd features for feature-mode (target-free)
calibration: each corner becomes an independent 3D landmark, triangulated from
the trajectory rather than read off a known layout.

Data-association soundness relies on the marker IDs being globally unique across
all boards (verified for the virig_4imus_3cams ArUco captures: DICT_6X6_250,
IDs 1..53, no in-frame duplicates on any camera).
"""

import logging

import numpy as np

from mvis.detection.aruco_board import aruco_detector
from mvis.detection.board import LazyDetector, scaled_to_original
from mvis.detection.frontend import FeatureFrontend
from mvis.core.types import Observation

logger = logging.getLogger(__name__)

_GEOMETRY_MSG = ("ArucoFeatureDetector provides features only — tags are treated as "
                 "unknown-3D landmarks, so board geometry is unavailable. Use "
                 "feature-mode (anonymous) calibration.")


class ArucoFeatureDetector(LazyDetector, FeatureFrontend):
    """ArUco detector exposing tag corners as geometry-free point features.

    NOT a board: the tags may span several physical boards at unknown relative
    poses, so no rigid layout is assumed (geometry methods raise). It subclasses
    ``Board`` only to satisfy MVIS's detection interface (`detect`)."""

    def __init__(
        self,
        aruco_dict: str = "DICT_6X6_250",
        *,
        min_tag_px: float = 20.0,
        max_scale: float = 4.0,
        refine_subpixel: bool = True,
    ):
        """
        Args:
            aruco_dict: ArUco dictionary name (e.g. ``DICT_6X6_250``).
            min_tag_px: if the median detected marker side is below this, the
                image is upscaled once and re-detected (helps small/distant tags).
            max_scale: cap on the upscale factor.
            refine_subpixel: enable OpenCV subpixel corner refinement
                (`CORNER_REFINE_SUBPIX`) — corner precision matters because each
                corner is a feature, not averaged into a board pose.
        """
        self._aruco_dict_name = aruco_dict
        self._min_tag_px = float(min_tag_px)
        self._max_scale = float(max_scale)
        self._refine = bool(refine_subpixel)
        self._detector = None  # lazy (cv2 import deferred)
        # in the board fingerprint, so detection caches written before the F6 upscale fix
        # (corners mapped back with x / scale) miss and re-detect
        self._corner_map = "pixel_centre"

    def _make_detector(self):
        """Build the cv2 ArucoDetector for the configured dictionary, with subpixel
        corner refinement when `refine_subpixel` (import deferred — optional dep)."""
        return aruco_detector(self._aruco_dict_name, self._refine)

    def _detect_raw(self, gray: np.ndarray):
        """Run marker detection, creating the detector lazily on first use."""
        corners, ids, _ = self._get_detector().detectMarkers(gray)
        return corners, ids

    @staticmethod
    def _median_side(corners_list) -> float:
        """Median marker side length (px) — the size signal for the upscale retry."""
        sizes = []
        for c in corners_list:
            p = c.reshape(4, 2)
            sizes.append(max(np.linalg.norm(p[i] - p[(i + 1) % 4]) for i in range(4)))
        return float(np.median(sizes))

    def detect(self, image: np.ndarray) -> list[Observation]:
        """Detect marker corners as point features: IDs are `marker_id*4 + corner`,
        retrying upscaled when markers are smaller than `min_tag_px` (size-based
        only — no grid-coverage assumption, unlike ArucoBoard). Rolling-shutter
        rows are set from the corner v-coordinate."""
        try:
            import cv2
        except ImportError:  # pragma: no cover
            raise ImportError(
                "ArUco detection requires 'opencv-contrib-python'. "
                "Install with: pip install opencv-contrib-python")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        corners, ids = self._detect_raw(gray)

        # Size-based upscale ONLY (no grid-coverage assumption, unlike ArucoBoard).
        scale = 1.0
        if ids is not None and len(ids):
            med = self._median_side(corners)
            if med < self._min_tag_px:
                scale = min(self._max_scale, max(1.5, self._min_tag_px / med * 1.5))
        elif ids is None:
            scale = 2.0
        if scale > 1.0:
            up = cv2.resize(gray, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_CUBIC)
            c2, i2 = self._detect_raw(up)
            n0 = 0 if ids is None else len(ids)
            if i2 is not None and len(i2) > n0:     # native corners win a tie
                logger.info("ArUco-feature upscale (%.1fx): %d -> %d markers",
                            scale, n0, len(i2))
                corners, ids = c2, i2
            else:
                scale = 1.0

        if ids is None:
            return []

        obs: list[Observation] = []
        for i, marker_id in enumerate(ids.flatten()):
            pts = corners[i].reshape(4, 2)
            if scale != 1.0:
                pts = scaled_to_original(pts, scale)
            for corner_idx in range(4):
                pid = int(marker_id) * 4 + corner_idx
                px = pts[corner_idx]
                obs.append(Observation(point_id=pid, pixel=px, row=int(px[1])))
        return obs

    # --- geometry withheld: tags are unknown-3D landmarks ---------------------
    def point_3d(self, point_id: int) -> np.ndarray:
        """Withheld — a FEATURE detector has no board geometry (see _GEOMETRY_MSG)."""
        raise NotImplementedError(_GEOMETRY_MSG)

    def all_points_3d(self) -> np.ndarray:
        """Withheld — a FEATURE detector has no board geometry (see _GEOMETRY_MSG)."""
        raise NotImplementedError(_GEOMETRY_MSG)

    def num_points(self) -> int:
        """Withheld — a FEATURE detector has no board geometry (see _GEOMETRY_MSG)."""
        raise NotImplementedError(_GEOMETRY_MSG)

    def min_points_for_pose(self) -> int:
        """Minimum detections for a usable pose — one full marker."""
        return 4
