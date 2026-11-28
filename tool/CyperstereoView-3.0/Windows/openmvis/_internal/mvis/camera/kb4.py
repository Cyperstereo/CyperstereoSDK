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
"""Kannala-Brandt camera model with 4 radial coefficients, theta from atan2.

theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8) with
theta = atan2(rho, Z), so it stays valid past 90 deg, up to the first fold of
theta_d(theta) (`kb_domain`). The same polynomial as `equidistant` (gtsam
Cal3Fisheye), which divides by Z first and so stops at 90 deg. Basalt's `kb4`.
Projection + unprojection in `custom_projections` (generic KB-N kernels). It exports to
Kalibr as pinhole-equi with the same k1..k4, exact for Z > 0; Kalibr cannot project past
90 deg (`cam_imu_output.kb4_kalibr_warning`).
"""

import numpy as np

from mvis.camera.camera_model import CameraModel, _kernel_project, _z_depth
from mvis.calibration.factors.custom_projections import project_kb, unproject_kb


class KannalaBrandt4(CameraModel):
    """Parameters: [fx, fy, cx, cy, k1, k2, k3, k4]. Exactly four coefficients
    (or none, meaning zeros): a longer list raises, it is never cut."""

    def __init__(self, fx, fy, cx, cy, dist_coeffs=None):
        self._fx, self._fy, self._cx, self._cy = (float(fx), float(fy),
                                                  float(cx), float(cy))
        d = (np.zeros(4) if dist_coeffs is None
             else np.asarray(dist_coeffs, dtype=np.float64).ravel())
        if len(d) != 4:
            raise ValueError(f"kb4 takes exactly 4 distortion coefficients "
                             f"[k1..k4], got {len(d)}")
        self._k = d.copy()

    def project(self, point_3d: np.ndarray, jac: bool = False):
        return _kernel_project(project_kb, np.asarray(point_3d, float),
                               self.cal_vec, self.n_intr, jac)

    def unproject(self, pixel: np.ndarray, depth: float = 1.0) -> np.ndarray:
        return _z_depth(unproject_kb(np.asarray(pixel, np.float64), self.cal_vec), depth)

    def get_gtsam_calibration(self):
        return None

    def intrinsics(self) -> np.ndarray:
        return np.array([self._fx, self._fy, self._cx, self._cy])

    def distortion(self) -> np.ndarray:
        return self._k.copy()

    def distortion_model_name(self) -> str:
        return "kb4"

    def num_params(self) -> int:
        return 8
