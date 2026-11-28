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
"""FOV camera model (Devernay & Faugeras 2001).

Single distortion parameter omega. Closed-form projection and unprojection.
Used by LSD-SLAM, DSO, Kalibr.
"""

import numpy as np

from mvis.camera.camera_model import CameraModel, _kernel_project, _z_depth
from mvis.calibration.factors.custom_projections import project_fov, unproject_fov


class FOVModel(CameraModel):
    """FOV distortion camera model.

    Parameters: [fx, fy, cx, cy, omega]
    omega: FOV parameter (radians). omega→0 reduces to pinhole.
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
            dist_coeffs = np.array([1.0])  # default omega
        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).ravel()
        self._omega = float(dist_coeffs[0]) if len(dist_coeffs) > 0 else 1.0
        self._cal_vec = np.array([fx, fy, cx, cy, self._omega, 0, 0, 0])

    def project(self, point_3d: np.ndarray, jac: bool = False):
        return _kernel_project(project_fov, np.asarray(point_3d, float),
                               self.cal_vec, self.n_intr, jac)

    def unproject(self, pixel: np.ndarray, depth: float = 1.0) -> np.ndarray:
        pixel = np.asarray(pixel, dtype=np.float64)
        return _z_depth(unproject_fov(pixel, self._cal_vec), depth)

    def get_gtsam_calibration(self):
        return None

    def intrinsics(self) -> np.ndarray:
        return np.array([self._fx, self._fy, self._cx, self._cy])

    def distortion(self) -> np.ndarray:
        return np.array([self._omega])

    def distortion_model_name(self) -> str:
        return "fov"

    def num_params(self) -> int:
        return 5
