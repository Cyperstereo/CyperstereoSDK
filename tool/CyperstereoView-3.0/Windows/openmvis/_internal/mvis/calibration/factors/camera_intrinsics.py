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
"""Camera calibration representation — GTSAM Cal3* ↔ flat 8-vector.

The camera analogue of `imu_intrinsics.py`: it holds the camera intrinsic
parameter representation. (Camera intrinsics carry no rotation, so a plain
8-vector suffices — no manifold class needed.) The factors take the
pixel-vs-intrinsics Jacobians from GTSAM's `Dcal`; a hand-written reference
lives in `tests/_reference/ref_intrinsic_jacobians.py` as a test oracle.
"""

from __future__ import annotations

import numpy as np
import gtsam

def _cal_from_vector(vec: np.ndarray, is_fisheye: bool):
    """Reconstruct GTSAM calibration from 8-element vector [fx,fy,cx,cy,k1,k2,k3/p1,k4/p2]
    (radtan / equidistant only — never a custom model's longer cal_vec)."""
    if len(vec) != 8:
        raise ValueError(f"_cal_from_vector needs an 8-element radtan/equidistant "
                         f"cal_vec, got {len(vec)}")
    fx, fy, cx, cy = vec[0], vec[1], vec[2], vec[3]
    if is_fisheye:
        return gtsam.Cal3Fisheye(fx, fy, 0.0, cx, cy, vec[4], vec[5], vec[6], vec[7])
    return gtsam.Cal3DS2(fx, fy, 0.0, cx, cy, vec[4], vec[5], vec[6], vec[7])


def _cal_to_vector(cal) -> np.ndarray:
    """Convert GTSAM calibration to 8-element vector [fx,fy,cx,cy,d1,d2,d3,d4].

    Cal3DS2 / Cal3Fisheye `vector()` is [fx, fy, s, cx, cy, d1..d4]; drop the skew s.
    """
    return np.delete(cal.vector(), 2)
