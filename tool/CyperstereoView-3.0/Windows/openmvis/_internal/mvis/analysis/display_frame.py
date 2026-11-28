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
"""The board display frame shared by the 3D views, and the equal-aspect cube around a point set.

The display transform rotates the board XY plane onto the display YZ plane, so the board stands
upright: (x_disp, y_disp, z_disp) = (z, x, y). For board points on z = 0 this is (0, x, y). It is
a pure permutation, used by the Plotly cam-only plots (plot_3d) and the Rerun viewer."""
from __future__ import annotations

import numpy as np

T_DISP = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])


def to_display_points(pts: np.ndarray) -> np.ndarray:
    """Apply the display permutation to an (N,3) array of points."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0:
        return pts
    out = np.empty_like(pts)            # [z, x, y]
    out[:, 0] = pts[:, 2]
    out[:, 1] = pts[:, 0]
    out[:, 2] = pts[:, 1]
    return out


def to_display_pose(T: np.ndarray) -> np.ndarray:
    """Apply the display transform to a 4x4 SE(3) matrix."""
    return T_DISP @ np.asarray(T, dtype=np.float64)


def cube_bounds(pts, min_half_range: float, pad: float = 1.0):
    """``(centre, half)`` of the equal-aspect cube around an (N,3) point set: the centre of its
    bounding box and the largest half-extent, floored at ``min_half_range``, times ``pad``.
    An empty set raises (each caller decides what an empty plot shows)."""
    pts = np.asarray(pts, float)
    lo, hi = pts.min(0), pts.max(0)
    return (lo + hi) / 2.0, max(float((hi - lo).max()) / 2.0, min_half_range) * pad
