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
"""Projection seam for the cam-IMU reprojection factors — model-agnostic.

The per-model camera math lives in `mvis.camera.CameraModel` (one class per
model: radtan/equidistant/fov/double_sphere/eucm/ucm/omni_radtan/kb4). This module is the
thin bridge the factors use: given a camera pose in world `T_CtoW` and a world
point `p_W`, produce the pixel + the Jacobians the factors need —

  * `J_pose`  (2x6, w.r.t. the T_CtoW right tangent),
  * `J_point` (2x3, w.r.t. the world point),
  * `J_intr`  (2 x n_intr, the model's intrinsics, our order).

It does this for EVERY model the same way: `CameraModel.project(p_cam, jac)` gives
the camera-frame `d(uv)/d(p_cam)` + `d(uv)/d(intr)`; the pose/point Jacobians are
chained through gtsam `Pose3.transformTo`'s `Dpose`/`Dpoint` (convention identical
to gtsam `PinholeCamera`, verified 1e-14). No `if model == ...` here — that lives
only in the `camera/` factory.
"""
from __future__ import annotations

import numpy as np
import gtsam

from mvis.camera.camera_model import build_camera_model


class _Projector:
    """Holds a CameraModel (built once per factor evaluation) and chains its
    camera-frame projection to world-point pose/point Jacobians."""
    __slots__ = ("model", "cal_vec", "_cm")

    def __init__(self, model: str, cal_vec: np.ndarray):
        self.model = model
        self.cal_vec = np.asarray(cal_vec, float)
        self._cm = build_camera_model(model, self.cal_vec)   # raises on unknown model

    def project(self, T_CtoW: gtsam.Pose3, p_W, jac: bool = True):
        """(uv[2], J_pose[2x6], J_point[2x3], J_intr[2 x n_intr]); jac=False ->
        (uv, None, None, None). Raises RuntimeError when the point is out of the
        model's range / behind the camera (the caller zeroes those rows)."""
        pt = gtsam.Point3(*np.asarray(p_W, float))
        if not jac:
            uv = self._cm.project(np.asarray(T_CtoW.transformTo(pt), float))
            if not np.isfinite(uv).all():
                raise RuntimeError(f"{self.model} projection out of model range")
            return uv, None, None, None
        D_pose = np.zeros((3, 6), order="F")
        D_point = np.zeros((3, 3), order="F")
        p_cam = T_CtoW.transformTo(pt, D_pose, D_point)
        uv, J_pc, J_intr = self._cm.project(np.asarray(p_cam, float), jac=True)
        if J_pc is None or not np.isfinite(uv).all():
            raise RuntimeError(f"{self.model} projection out of model range")
        return uv, J_pc @ D_pose, J_pc @ D_point, J_intr


def make_projector(model: str, cal_vec: np.ndarray) -> _Projector:
    """Build a projector for `model` (call once per factor eval, reuse per corner)."""
    return _Projector(model, cal_vec)
