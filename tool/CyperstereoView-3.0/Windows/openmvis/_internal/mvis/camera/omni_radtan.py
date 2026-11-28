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
"""Omnidirectional (MEI) camera model with radial-tangential distortion.

A unit-sphere projection (mirror parameter xi) followed by radtan distortion on
the normalized plane. Closed-form projection + unprojection in
`custom_projections`. Used by some wide-FOV / catadioptric rigs (Kalibr `omni`).
"""

import numpy as np

from mvis.camera.camera_model import CameraModel, _kernel_project, _z_depth
from mvis.calibration.factors.custom_projections import (
    project_omni_radtan, unproject_omni_radtan,
)


class OmniRadtan(CameraModel):
    """Omni + radial-tangential model.

    Parameters: [fx, fy, cx, cy, xi, k1, k2, p1, p2]  (Kalibr omni-radtan)
    xi >= 0: mirror/sphere parameter; k1,k2 radial; p1,p2 tangential.
    Legacy 4-coeff input [xi, k1, k2, p1] loads with p2 = 0.
    """

    def __init__(self, fx, fy, cx, cy, dist_coeffs=None):
        self._fx, self._fy, self._cx, self._cy = (float(fx), float(fy),
                                                  float(cx), float(cy))
        if dist_coeffs is None:
            dist_coeffs = np.array([1.0, 0.0, 0.0, 0.0, 0.0])   # xi, k1, k2, p1, p2
        d = np.asarray(dist_coeffs, dtype=np.float64).ravel()
        if len(d) > 5:
            raise ValueError(f"omni_radtan takes 5 distortion coefficients "
                             f"[xi, k1, k2, p1, p2], got {len(d)}")
        if len(d) < 5:
            d = np.pad(d, (0, 5 - len(d)))      # legacy [xi, k1, k2, p1] -> p2 = 0
        self._xi, self._k1, self._k2, self._p1, self._p2 = (
            float(d[0]), float(d[1]), float(d[2]), float(d[3]), float(d[4]))

    def project(self, point_3d: np.ndarray, jac: bool = False):
        return _kernel_project(project_omni_radtan, np.asarray(point_3d, float),
                               self.cal_vec, self.n_intr, jac)

    def unproject(self, pixel: np.ndarray, depth: float = 1.0) -> np.ndarray:
        return _z_depth(unproject_omni_radtan(np.asarray(pixel, np.float64), self.cal_vec), depth)

    def get_gtsam_calibration(self):
        return None

    def intrinsics(self) -> np.ndarray:
        return np.array([self._fx, self._fy, self._cx, self._cy])

    def distortion(self) -> np.ndarray:
        return np.array([self._xi, self._k1, self._k2, self._p1, self._p2])

    def distortion_model_name(self) -> str:
        return "omni_radtan"

    def num_params(self) -> int:
        return 9
