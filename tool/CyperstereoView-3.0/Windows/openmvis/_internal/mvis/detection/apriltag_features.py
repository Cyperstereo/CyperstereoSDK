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
"""AprilTag tags as stable point features (geometry-free front-end).

AprilTag analog of :class:`ArucoFeatureDetector`: emits every detected tag's 4
corners as :class:`Observation`s keyed by ``point_id = tag_id*4 + corner`` with
**no grid ceiling** and **no board geometry**. For feature-mode (target-free)
calibration where AprilTags are used purely as uniquely-ID'd features (e.g. the
mvis_4imus_4cams target). Data-association soundness relies on the AprilTag IDs
being globally unique across cameras (true for a single AprilGrid family).
"""

import logging

import numpy as np

from mvis.detection.board import LazyDetector, scaled_to_original
from mvis.detection.frontend import FeatureFrontend
from mvis.core.types import Observation

logger = logging.getLogger(__name__)

_GEOMETRY_MSG = ("AprilTagFeatureDetector provides features only — tags are treated "
                 "as unknown-3D landmarks, so board geometry is unavailable. "
                 "Use feature-mode (anonymous) calibration.")


class AprilTagFeatureDetector(LazyDetector, FeatureFrontend):
    """AprilTag detector exposing tag corners as geometry-free point features."""

    _MODULES_PER_SIDE = 10            # t36h11: 8 data + 2-bit border

    def __init__(self, tag_family: str = "t36h11", *,
                 min_tag_px: float = 30.0, max_scale: float = 4.0,
                 refine_subpixel: bool = True):
        """`min_tag_px`/`max_scale` bound the size-based upscale retry in `detect`;
        `refine_subpixel` enables the cv2 corner refinement pass."""
        self._tag_family = tag_family
        self._min_tag_px = float(min_tag_px)
        self._max_scale = float(max_scale)
        self._refine = bool(refine_subpixel)
        self._detector = None
        # in the board fingerprint, so detection caches written before the F6 upscale fix
        # (corners mapped back with x / scale) miss and re-detect
        self._corner_map = "pixel_centre"

    def _make_detector(self):
        """The aprilgrid detector, built once on first use (optional dependency)."""
        try:
            import aprilgrid
        except ImportError:  # pragma: no cover
            raise ImportError("AprilTag detection requires 'aprilgrid'. "
                              "Install with: pip install aprilgrid")
        return aprilgrid.Detector(self._tag_family)

    def _detect_raw(self, gray):
        """Run the raw detector, mapping aprilgrid's internal cornerSubPix assert
        (a cv2.error) to "no detections" instead of crashing the frame."""
        import cv2
        try:
            return self._get_detector().detect(gray)
        except cv2.error:        # aprilgrid's internal cornerSubPix can assert
            return []

    @staticmethod
    def _median_side(dets) -> float:
        """Median tag side length (px) across detections — the size signal that
        drives the upscale retry."""
        sizes = []
        for d in dets:
            c = np.asarray(d.corners).reshape(-1, 2)
            sizes.append(float(np.median(
                [np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)])))
        return float(np.median(sizes)) if sizes else 0.0

    def detect(self, image: np.ndarray) -> list[Observation]:
        """Detect tag corners as point features: IDs are `tag_id*4 + corner`,
        retrying at an upscaled resolution when tags are smaller than
        `min_tag_px`. Returns Observations with rolling-shutter rows set."""
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        dets = self._detect_raw(gray)
        scale = 1.0
        # size-based upscale only (tag side in px vs min)
        if dets:
            med = self._median_side(dets)
            if med < self._min_tag_px:
                scale = min(self._max_scale, max(1.5, self._min_tag_px / max(med, 1.0)))
        else:
            scale = 2.0
        if scale > 1.0:
            up = cv2.resize(gray, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_CUBIC)
            d2 = self._detect_raw(up)
            if len(d2) > len(dets):                 # native corners win a tie
                logger.info("AprilTag-feature upscale (%.1fx): %d -> %d tags",
                            scale, len(dets), len(d2))
                dets, _scale_used = d2, scale
            else:
                scale = 1.0
        else:
            _scale_used = 1.0

        if not dets:
            return []

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        obs: list[Observation] = []
        for d in dets:
            corners = np.asarray(d.corners, float).reshape(-1, 2)
            if scale != 1.0:
                corners = scaled_to_original(corners, scale)
            if self._refine:
                edge = float(np.linalg.norm(corners[1] - corners[0]))
                win = max(3, int(min(edge / 4.0, 15)))
                pts = corners.reshape(-1, 1, 2).astype(np.float32)
                try:
                    cv2.cornerSubPix(gray, pts, (win, win), (-1, -1), criteria)
                    corners = pts.reshape(-1, 2).astype(np.float64)
                except cv2.error:
                    pass
            for ci in range(4):
                pid = int(d.tag_id) * 4 + ci
                px = corners[ci]
                obs.append(Observation(point_id=pid, pixel=px, row=int(px[1])))
        return obs

    # --- geometry withheld ---------------------------------------------------
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
        """Minimum detections for a usable pose — one full tag."""
        return 4
