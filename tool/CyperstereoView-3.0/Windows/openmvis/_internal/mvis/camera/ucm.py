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
"""Unified Camera Model (Geyer & Daniilidis 2000; Mei & Rives 2007), alpha form.

Parameters [fx, fy, cx, cy, alpha]: the EUCM with beta = 1 (Basalt `ucm`). Kalibr
`omni` with distortion `none` is the same model in xi form:
xi = alpha / (1 - alpha), gamma = fx / (1 - alpha) (converted at IO only).
Closed-form projection and unprojection (`custom_projections.project_ucm`).
"""

import numpy as np

from mvis.camera.camera_model import CameraModel, _kernel_project, _z_depth
from mvis.calibration.factors.custom_projections import project_ucm, unproject_ucm
from mvis.core.camera_models import canonical_dist


class UCM(CameraModel):
    """Unified Camera Model.

    Parameters: [fx, fy, cx, cy, alpha]
    alpha ∈ [0, 1): blend (0=pinhole); xi = alpha / (1 - alpha)
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
            dist_coeffs = [0.6]  # default
        # one coefficient; a longer list only when the extras are zero
        self._alpha = float(canonical_dist("ucm", np.asarray(dist_coeffs, float).ravel())[0])
        self._cal_vec = np.array([fx, fy, cx, cy, self._alpha, 0, 0, 0])

    def project(self, point_3d: np.ndarray, jac: bool = False):
        return _kernel_project(project_ucm, np.asarray(point_3d, float),
                               self.cal_vec, self.n_intr, jac)

    def unproject(self, pixel: np.ndarray, depth: float = 1.0) -> np.ndarray:
        pixel = np.asarray(pixel, dtype=np.float64)
        return _z_depth(unproject_ucm(pixel, self._cal_vec), depth)

    def get_gtsam_calibration(self):
        return None

    def intrinsics(self) -> np.ndarray:
        return np.array([self._fx, self._fy, self._cx, self._cy])

    def distortion(self) -> np.ndarray:
        return np.array([self._alpha])

    def distortion_model_name(self) -> str:
        return "ucm"

    def num_params(self) -> int:
        return 5

    def xi(self) -> float:
        """Kalibr / Mei xi = alpha / (1 - alpha)."""
        return self._alpha / (1.0 - self._alpha)

    def kalibr_gamma(self) -> tuple[float, float]:
        """Kalibr omni focal (gamma_u, gamma_v) = (fx, fy) / (1 - alpha)."""
        return self._fx / (1.0 - self._alpha), self._fy / (1.0 - self._alpha)
