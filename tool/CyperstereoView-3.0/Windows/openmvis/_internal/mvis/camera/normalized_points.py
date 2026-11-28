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
"""Pixels to normalized image points (x/z, y/z), one entry point for the init paths.

Two backends:

* ``"camera_model"`` (default): the model's own inverse, ``CameraModel.unproject_to_normalized``,
  for every init path (camera-only initializer, mono SfM, cam-IMU PnP track);
* ``"cv2"``: OpenCV ``undistortPoints`` with K and the radtan coefficients (radtan only;
  every point is valid), mono SfM's radtan path.

``max_ray_deg`` additionally drops rays further off-axis than that angle (the initializer's
PnP cap).
"""

from __future__ import annotations

import numpy as np


def normalized_points(pixels, model: str, cal_vec, backend: str = "camera_model",
                      max_ray_deg: float | None = None):
    """Pixels (N, 2) -> (xy (N, 2), valid (N,) bool) on the normalized plane.

    `cal_vec` is [fx, fy, cx, cy, dist...]. Rows that are not valid hold zeros; use the mask.
    """
    cal_vec = np.asarray(cal_vec, float)
    if backend == "cv2":
        import cv2
        from mvis.core.camera_models import canonical_dist
        if model != "radtan":
            raise ValueError(f"cv2 backend is radtan only, got {model!r}")
        if max_ray_deg is not None:
            raise ValueError("max_ray_deg needs the camera_model backend")
        K = np.array([[cal_vec[0], 0, cal_vec[2]],
                      [0, cal_vec[1], cal_vec[3]],
                      [0, 0, 1.0]])
        dist = np.asarray(canonical_dist(model, cal_vec[4:]))
        uv = np.asarray(pixels, np.float64).reshape(-1, 1, 2)
        xy = cv2.undistortPoints(uv, K, dist).reshape(-1, 2)
        return xy, np.ones(len(xy), bool)
    if backend != "camera_model":
        raise ValueError(f"unknown backend {backend!r}")
    from mvis.camera.camera_model import build_camera_model
    xy, valid = build_camera_model(model, cal_vec).unproject_to_normalized(pixels)
    if max_ray_deg is not None:     # cos of the off-axis angle of the ray (x, y, 1)
        cos_t = 1.0 / np.linalg.norm(np.c_[xy, np.ones(len(xy))], axis=1)
        valid &= cos_t > np.cos(np.deg2rad(max_ray_deg))
    return xy, valid
