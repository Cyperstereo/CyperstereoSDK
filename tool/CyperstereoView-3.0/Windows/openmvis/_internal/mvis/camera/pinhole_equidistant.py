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
"""Pinhole camera model with equidistant (fisheye) distortion.

Wraps GTSAM Cal3Fisheye: (fx, fy, s, cx, cy, k1, k2, k3, k4). It divides by z first, so its domain
ends at z = 0, 90 deg off-axis: past that a corner has no pixel here, and `kb4` is the model for it.
The inverse is unique only up to the polynomial's first fold, which can come before 90 deg.
"""

import numpy as np
import gtsam

from mvis.calibration.factors.custom_projections import unproject_kb
from mvis.camera.camera_model import CameraModel, _gtsam_project, _in_front, _z_depth


class PinholeEquidistant(CameraModel):
    """Pinhole camera with equidistant/fisheye distortion.

    Distortion model:
        theta = atan2(r, 1) where r = sqrt(x_n^2 + y_n^2)
        theta_d = theta*(1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
        x_d = (theta_d / r) * x_n
        y_d = (theta_d / r) * y_n

    Parameters: [k1, k2, k3, k4]
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        dist_coeffs: np.ndarray | None = None,
    ):
        if dist_coeffs is None:
            dist_coeffs = np.zeros(4)
        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)
        if len(dist_coeffs) < 4:
            dist_coeffs = np.pad(dist_coeffs, (0, 4 - len(dist_coeffs)))

        k1, k2, k3, k4 = dist_coeffs[:4]
        self._cal = gtsam.Cal3Fisheye(fx, fy, 0.0, cx, cy, k1, k2, k3, k4)
        self._dist_coeffs = dist_coeffs[:4].copy()

    def project(self, point_3d: np.ndarray, jac: bool = False):
        point_3d = np.asarray(point_3d, dtype=np.float64)
        if jac:
            return _gtsam_project(self._cal, point_3d, True)
        if not _in_front(point_3d):     # the division below would mirror it into the image
            return np.array([np.nan, np.nan])
        try:
            p_norm = point_3d[:2] / point_3d[2]
            p2d = self._cal.uncalibrate(p_norm)
            return np.array([p2d[0], p2d[1]])
        except RuntimeError:
            return np.array([np.nan, np.nan])

    def unproject(self, pixel: np.ndarray, depth: float = 1.0) -> np.ndarray:
        """The ray of `pixel` on the first branch of the radial polynomial, the only one with a
        unique inverse: the same root as `kb4` for the same coefficients, up to the first fold or
        90 deg, NaN past that. Not gtsam's `calibrate`: its Newton solve can report success on a
        later branch, whose ray reprojects to the same pixel, can fail on a valid edge pixel, and
        stops about 1e-5 px short."""
        pixel = np.asarray(pixel, dtype=np.float64)
        if not (np.isfinite(pixel).all() and np.isfinite(depth)
                and np.isfinite(self._dist_coeffs).all()):
            return np.full(3, np.nan)
        return _z_depth(unproject_kb(pixel, np.r_[self.intrinsics(), self._dist_coeffs]), depth)

    def get_gtsam_calibration(self) -> gtsam.Cal3Fisheye:
        return self._cal

    def intrinsics(self) -> np.ndarray:
        return np.array([self._cal.fx(), self._cal.fy(),
                         self._cal.px(), self._cal.py()])

    def distortion(self) -> np.ndarray:
        return self._dist_coeffs.copy()

    def distortion_model_name(self) -> str:
        return "equidistant"

    def num_params(self) -> int:
        return 8  # fx, fy, cx, cy, k1, k2, k3, k4

    def to_opencv(self) -> tuple[np.ndarray, np.ndarray]:
        """Return OpenCV-compatible camera matrix and distortion coefficients.

        Note: OpenCV fisheye uses cv2.fisheye functions, not the standard
        cv2.calibrateCamera distortion model.
        """
        K = np.array([
            [self._cal.fx(), 0, self._cal.px()],
            [0, self._cal.fy(), self._cal.py()],
            [0, 0, 1],
        ])
        dist = self._dist_coeffs.copy()
        return K, dist
