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
"""detection.quality — the component's detection-quality facade: the
`frame_quality` metrics and the `quality_gate` that feed the frame selection
flags, re-exported from their implementation modules so that user code can
``from mvis.detection.quality import frame_quality, quality_gate``."""
from mvis.detection import frame_quality, quality_gate           # noqa: F401

__all__ = ["frame_quality", "quality_gate"]
