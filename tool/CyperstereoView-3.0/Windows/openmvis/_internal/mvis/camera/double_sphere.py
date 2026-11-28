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
"""Double Sphere camera model (Usenko et al. 2018).

Closed-form projection AND unprojection. Fastest Jacobians.
Default model in Basalt VIO. Recommended for fisheye/wide-angle cameras.
"""

import numpy as np

from mvis.camera.camera_model import CameraModel, _kernel_project, _z_depth
from mvis.calibration.factors.custom_projections import (
    project_double_sphere, unproject_double_sphere,
)


class DoubleSphere(CameraModel):
    """Double Sphere camera model.

    Parameters: [fx, fy, cx, cy, xi, alpha]
    xi ∈ [-1, 1]: first sphere parameter
    alpha ∈ [0, 1]: second sphere parameter
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        dist_coeffs: np.ndarray | None = None,
    ):
        self._fx = float(fx)
        self._fy = float(fy)
        self._cx = float(cx)
        self._cy = float(cy)
        if dist_coeffs is None:
            dist_coeffs = np.array([-0.2, 0.6])  # Basalt defaults
        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).ravel()
        self._xi = float(dist_coeffs[0]) if len(dist_coeffs) > 0 else -0.2
        self._alpha = float(dist_coeffs[1]) if len(dist_coeffs) > 1 else 0.6
        self._cal_vec = np.array([fx, fy, cx, cy, self._xi, self._alpha, 0, 0])

    def project(self, point_3d: np.ndarray, jac: bool = False):
        return _kernel_project(project_double_sphere, np.asarray(point_3d, float),
                               self.cal_vec, self.n_intr, jac)

    def unproject(self, pixel: np.ndarray, depth: float = 1.0) -> np.ndarray:
        pixel = np.asarray(pixel, dtype=np.float64)
        return _z_depth(unproject_double_sphere(pixel, self._cal_vec), depth)

    def get_gtsam_calibration(self):
        return None

    def intrinsics(self) -> np.ndarray:
        return np.array([self._fx, self._fy, self._cx, self._cy])

    def distortion(self) -> np.ndarray:
        return np.array([self._xi, self._alpha])

    def distortion_model_name(self) -> str:
        return "double_sphere"

    def num_params(self) -> int:
        return 6
