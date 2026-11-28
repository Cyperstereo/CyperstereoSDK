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
"""Rotation-matrix helpers for reports and comparisons; solver code has its own (factors/lie_groups)."""
from __future__ import annotations

import numpy as np


def rotation_angle(R) -> float:
    """Geodesic angle |log(R)| in radians. atan2(sin, cos) from the skew part keeps full resolution
    for tiny angles (arccos of the trace bottoms out at ~1.5e-8 rad from the 1e-16 rounding of the
    trace) and near pi. For the distance between two rotations pass Ra.T @ Rb."""
    R = np.asarray(R, float)
    s = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    c = (np.trace(R) - 1.0) * 0.5
    return float(np.arctan2(np.linalg.norm(s), c))
