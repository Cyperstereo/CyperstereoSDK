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
"""Pinhole camera model with radial-tangential distortion.

Wraps GTSAM Cal3DS2: (fx, fy, s, cx, cy, k1, k2, p1, p2).
"""

import numpy as np
import gtsam

from mvis.camera.camera_model import CameraModel, _gtsam_project, _in_front


class PinholeRadtan(CameraModel):
    """Pinhole camera with radial-tangential distortion.

    Distortion model (applied to normalized coordinates):
        r^2 = x_n^2 + y_n^2
        x_d = x_n*(1 + k1*r^2 + k2*r^4) + 2*p1*x_n*y_n + p2*(r^2 + 2*x_n^2)
        y_d = y_n*(1 + k1*r^2 + k2*r^4) + p1*(r^2 + 2*y_n^2) + 2*p2*x_n*y_n

    Parameters: [k1, k2, p1, p2]
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

        k1, k2, p1, p2 = dist_coeffs[:4]
        self._cal = gtsam.Cal3DS2(fx, fy, 0.0, cx, cy, k1, k2, p1, p2)
        self._dist_coeffs = dist_coeffs[:4].copy()

    def project(self, point_3d: np.ndarray, jac: bool = False):
        point_3d = np.asarray(point_3d, dtype=np.float64)
        if jac:
            return _gtsam_project(self._cal, point_3d, True)
        if not _in_front(point_3d):     # behind the camera: NaN, as the jac=True path gives
            return np.array([np.nan, np.nan])
        try:
            p2d = self._cal.uncalibrate(point_3d[:2] / point_3d[2])
            return np.array([p2d[0], p2d[1]])
        except RuntimeError:
            return np.array([np.nan, np.nan])

    def unproject(self, pixel: np.ndarray, depth: float = 1.0) -> np.ndarray:
        pixel = np.asarray(pixel, dtype=np.float64)
        p2d = gtsam.Point2(pixel[0], pixel[1])
        p_cal = self._cal.calibrate(p2d)
        return np.array([p_cal[0] * depth, p_cal[1] * depth, depth])

    def get_gtsam_calibration(self) -> gtsam.Cal3DS2:
        return self._cal

    def intrinsics(self) -> np.ndarray:
        return np.array([self._cal.fx(), self._cal.fy(),
                         self._cal.px(), self._cal.py()])

    def distortion(self) -> np.ndarray:
        return self._dist_coeffs.copy()

    def distortion_model_name(self) -> str:
        return "radtan"

    def num_params(self) -> int:
        return 8  # fx, fy, cx, cy, k1, k2, p1, p2

    def to_opencv(self) -> tuple[np.ndarray, np.ndarray]:
        """Return OpenCV-compatible camera matrix and distortion coefficients."""
        K = np.array([
            [self._cal.fx(), 0, self._cal.px()],
            [0, self._cal.fy(), self._cal.py()],
            [0, 0, 1],
        ])
        dist = np.array([self._dist_coeffs[0], self._dist_coeffs[1],
                         self._dist_coeffs[2], self._dist_coeffs[3], 0.0])
        return K, dist

    @classmethod
    def from_opencv(cls, K: np.ndarray, dist_coeffs: np.ndarray) -> "PinholeRadtan":
        """Create from OpenCV camera matrix and distortion coefficients."""
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        d = np.asarray(dist_coeffs).flatten()
        k1 = d[0] if len(d) > 0 else 0.0
        k2 = d[1] if len(d) > 1 else 0.0
        p1 = d[2] if len(d) > 2 else 0.0
        p2 = d[3] if len(d) > 3 else 0.0
        return cls(fx, fy, cx, cy, np.array([k1, k2, p1, p2]))
