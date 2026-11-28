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
"""detection — the detection component.

Targets are data; detectors are algorithms:
  Board (mvis.detection.board, geometry)  <-  what db.boards stores
  FeatureDetector hierarchy        <-  the algorithms, composable by mode
  sweep(source, detectors, db)     <-  every frame, every enabled detector,
                                       filling the frozen measurement core

This package is the only home of targets and detectors; every call site
imports mvis.detection directly.
"""
from mvis.detection.detectors import (
    BoardFeatureDetector, FeatureDetector, TagFeatureDetector,
)
from mvis.detection.frontend import FeatureFrontend
from mvis.detection.sweep import sweep

__all__ = ["FeatureDetector", "FeatureFrontend", "BoardFeatureDetector",
           "TagFeatureDetector", "sweep"]
