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
"""CameraCalib — per-camera calibration state, a Python product manifold.

The camera analogue of `ImuCalib` (`imu_calib.py`): it bundles all per-camera
calibration into one cohesive (8 + n_intr)-dim product manifold, so the joint camera-IMU
solve can expose a full per-camera calibration covariance via gtsam marginals,
the same way `ImuCalib` does for the IMU side.

    T_CtoI : Pose3     camera->IMU extrinsic        (6)
    td     : scalar    camera-IMU time offset       (1)
    tr     : scalar    rolling-shutter readout      (1)
    intr   : Vector    [fx,fy,cx,cy,dist...]        (n_intr)

As with `ImuCalib`, gtsam-Python `Values` cannot hold a custom manifold type, so
in the factor graph this is stored as its (8 + n_intr)-dim minimal-tangent `Vector` ξ
around a fixed base estimate, and the factor evaluates `base.retract(ξ)` (ξ is a
true vector coordinate, so gtsam's additive update on it is the correct manifold
step). `is_fisheye` is fixed metadata that rides along — it selects the Cal3
model, it is not an optimized coordinate.

Only `T_CtoI` is rotation-bearing, so only the first 6 tangent entries use
`Pose3.retract`; td, tr, and the (Euclidean) intrinsics update additively. This
mirrors the rule used throughout the port: only rotation-bearing state needs
manifold treatment.

Tangent layout (rotation-bearing pose first, then the temporal scalars, then the
Euclidean intrinsics — the camera analogue of `ImuCalib::retract`):
    ξ = [ d_pose(6) | d_td(1) | d_tr(1) | d_intr(n_intr) ]
"""

from __future__ import annotations

import numpy as np
import gtsam

from mvis.calibration.factors.camera_intrinsics import (
    _cal_from_vector, _cal_to_vector,
)
from mvis.core.camera_models import (
    GTSAM, canonical_cal_vec, check_model, legacy_model, n_dist,
)


class CameraCalib:
    """Per-camera calib bundle. Manifold dim is per-instance: 8 + len(intr)
    (= pose6 + td1 + tr1 + n_intr); `model` (str) selects the projection and the
    intrinsics length (4 + `core.camera_models.n_dist`) — radtan/equidistant = 8,
    FOV/ucm = 5, double_sphere/eucm = 6, omni_radtan = 9, kb4 = 8; a padded or
    legacy intr is canonicalized to that length. `is_fisheye` is kept as a back-compat alias for
    `model == "equidistant"`."""

    def __init__(self, T_CtoI: gtsam.Pose3, td, tr, intr,
                 is_fisheye: bool = False, model: str | None = None):
        self.T = T_CtoI
        self.td = float(np.asarray(td).ravel()[0])
        self.tr = float(np.asarray(tr).ravel()[0])
        # model is canonical; is_fisheye kept for back-compat (equidistant only)
        self.intr = np.asarray(intr, float).copy()   # [fx,fy,cx,cy, dist...]
        self.model = check_model(str(model) if model is not None
                                 else legacy_model(is_fisheye, self.intr))
        if self.model == "kb4" and len(self.intr) != 8:
            raise ValueError(f"kb4 intr must be 8 long, got {len(self.intr)}")
        if len(self.intr) != 4 + n_dist(self.model):
            # legacy / padded intr -> the model's canonical length (or raise)
            self.intr = np.asarray(canonical_cal_vec(self.model, self.intr), float)

    @property
    def DIM(self) -> int:
        """Tangent dim = pose(6) + td(1) + tr(1) + intrinsics(n_intr)."""
        return 8 + len(self.intr)

    @property
    def n_intr(self) -> int:
        """Number of intrinsic parameters (model-dependent)."""
        return len(self.intr)

    @property
    def is_fisheye(self) -> bool:
        """Back-compat alias: equidistant is the only fisheye/gtsam-Cal3Fisheye model."""
        return self.model == "equidistant"

    @classmethod
    def from_cal(cls, T_CtoI, td, tr, cal, is_fisheye=None):
        """Build from a gtsam Cal3DS2/Cal3Fisheye instead of an 8-vector.

        is_fisheye defaults to the runtime type of `cal`.
        """
        if is_fisheye is None:
            is_fisheye = isinstance(cal, gtsam.Cal3Fisheye)
        return cls(T_CtoI, td, tr, _cal_to_vector(cal), is_fisheye)

    # --- accessors used by camera-IMU coupling ---
    @property
    def R_CtoI(self) -> np.ndarray:
        """Rotation camera->IMU: the 3x3 R of T_CtoI."""
        return self.T.rotation().matrix()

    @property
    def p_CinI(self) -> np.ndarray:
        """Camera origin in the IMU frame: the 3-vector t of T_CtoI (m)."""
        return np.asarray(self.T.translation(), float)

    @property
    def cal(self):
        """The gtsam Cal3DS2/Cal3Fisheye for the current intrinsics (radtan /
        equidistant only — use `camera_model` for the others)."""
        if self.model not in GTSAM:
            raise ValueError(f"CameraCalib.cal: no gtsam Cal3 for model {self.model!r}")
        return _cal_from_vector(self.intr, self.is_fisheye)

    @property
    def camera_model(self):
        """The `camera.CameraModel` for this camera's model + current intrinsics —
        the single object that knows how to project/unproject for every model."""
        from mvis.camera.camera_model import build_camera_model
        return build_camera_model(self.model, self.intr)

    # --- manifold ops (dim = 8 + n_intr) ---
    def retract(self, xi: np.ndarray) -> "CameraCalib":
        """Manifold retract: apply tangent xi (6 pose + td + tr + n_intr) -> a new CameraCalib."""
        xi = np.asarray(xi, float)
        n = len(self.intr)
        return CameraCalib(
            self.T.retract(xi[0:6]),
            self.td + xi[6],
            self.tr + xi[7],
            self.intr + xi[8:8 + n],
            model=self.model)

    def local_coordinates(self, other: "CameraCalib") -> np.ndarray:
        """Manifold log: tangent from self to other (inverse of retract)."""
        return np.concatenate([
            self.T.localCoordinates(other.T),
            [other.td - self.td],
            [other.tr - self.tr],
            other.intr - self.intr])

    def copy(self) -> "CameraCalib":
        """Deep copy (independent Pose3 + intrinsics array)."""
        return CameraCalib(gtsam.Pose3(self.T.matrix()), self.td, self.tr,
                           self.intr.copy(), model=self.model)

    def __repr__(self):
        return (f"CameraCalib(model={self.model}, td={self.td:.6f}, "
                f"tr={self.tr:.6f}, p_CinI={self.p_CinI}, fx={self.intr[0]:.2f})")
