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
"""FeatureDetector hierarchy: the general detection classes.

The name history matters: the old base class `Board` conflated the physical
target with the detection algorithm — the geometry-free feature detectors
subclass it and RAISE on geometry access. Here the split is explicit:

  FeatureDetector           the algorithm contract: detect(image, t) per frame
  BoardFeatureDetector      wraps a geometry-full board target (AprilGrid,
                            ArUco grid, chessboard): known layout, PnP-able
  TagFeatureDetector        wraps the geometry-free *_features detectors:
                            ID'd corners, NO coplanarity assumed
  (FreeFeatureDetector)     future: anonymous tracks; STATEFUL — the sweep
                            feeds frames in time order per camera for it

Both wrappers delegate to the mvis.detection implementations so behavior is
byte-identical while call sites migrate.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from mvis.mvisdb.mvisdatabase import DetectionMode, TagDetection


class FeatureDetector(ABC):
    """One frame in -> detections out. Composable: a capture may run several
    detectors of different modes over the same image."""

    mode: DetectionMode

    @abstractmethod
    def detect(self, image: np.ndarray, timestamp: float = 0.0) -> list[TagDetection]:
        """Detections for THIS frame, in detector emission order."""


def _to_tag_detections(observations) -> list[TagDetection]:
    """Legacy Observation list -> TagDetection list, VERBATIM: uv float64
    untouched, point_id as emitted, emission order = list index (the
    bit-identity replay column)."""
    return [TagDetection(uv=np.asarray(o.pixel, float), point_id=int(o.point_id),
                         detector_emission_order=k)
            for k, o in enumerate(observations)]


class BoardFeatureDetector(FeatureDetector):
    """A detector for one geometry-full board target.

    Wraps a legacy board target (which still carries detect()); the
    geometry half of that object is what bridge.board_from_geometry turns into
    the stored Board entity."""

    mode = DetectionMode.BOARD

    def __init__(self, board, board_id: int = 0):
        self.board = board
        self.board_id = board_id

    def detect(self, image: np.ndarray, timestamp: float = 0.0) -> list[TagDetection]:
        return _to_tag_detections(self.board.detect(image))


class TagFeatureDetector(FeatureDetector):
    """Geometry-free ID'd corners (the already-existing *_features detectors:
    ArucoFeatureDetector / AprilTagFeatureDetector). No board layout is
    assumed; 3D belongs to the run's landmarks, not the target."""

    mode = DetectionMode.TAG

    def __init__(self, feature_detector):
        self.impl = feature_detector

    def detect(self, image: np.ndarray, timestamp: float = 0.0) -> list[TagDetection]:
        return _to_tag_detections(self.impl.detect(image))
