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
"""FeatureFrontend — the contract feature-mode detection actually has.

The geometry-free feature detectors
(AprilTagFeatureDetector, ArucoFeatureDetector) used to subclass Board and
implement its geometry methods only to raise. Their REAL contract, as consumed
by every pipeline, is exactly this: detect a frame, and say how many points a
pose needs. 3-D for their features belongs to the run's landmarks,
never to a target.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class FeatureFrontend(ABC):
    """Geometry-free per-frame feature detection."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list:
        """Observations for this frame, in emission order."""

    @abstractmethod
    def min_points_for_pose(self) -> int:
        """Minimum observations for a usable frame (ingest keep-filter)."""
