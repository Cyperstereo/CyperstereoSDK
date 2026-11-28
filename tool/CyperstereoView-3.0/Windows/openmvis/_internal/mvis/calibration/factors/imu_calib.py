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
"""ImuCalib — auxiliary-IMU calibration state, a Python product manifold.

Mirrors the C++ MVIS `graph_Nodes/ImuCalib` (a registered gtsam Manifold in C++). The
state is a 31-dim product manifold:

    T_IstoI : Pose3          aux->base extrinsic   (6)
    td      : scalar         time offset           (1)
    imu_in  : ImuIntrinsics  Dw, Da, Tg, R         (24)

gtsam-Python `Values` cannot hold a custom manifold type, so in the factor graph
this is stored as its 31-dim minimal-tangent `Vector` ξ around a fixed base
estimate, and the factor evaluates `base.retract(ξ)` (ξ is a true vector
coordinate, so gtsam's additive update on it is the correct manifold step).

Tangent layout (matches C++ ImuCalib::retract):
    ξ = [ d_pose(6) | d_td(1) | d_imu_in(24) ]
"""

from __future__ import annotations

import numpy as np
import gtsam

from mvis.imu import ImuModel


class ImuCalib:
    DIM = 31

    def __init__(self, T_IstoI: gtsam.Pose3, td, imu_in: ImuModel):
        self.T = T_IstoI
        self.td = float(np.asarray(td).ravel()[0])
        self.imu_in = imu_in

    # --- accessors used by the factor's kinematic coupling ---
    @property
    def R_IstoI(self) -> np.ndarray:
        """Rotation aux-IMU->base-IMU: the 3x3 R of T_IstoI."""
        return self.T.rotation().matrix()

    @property
    def p_IsinI(self) -> np.ndarray:
        """Aux-IMU origin in the base-IMU frame: the 3-vector t of T_IstoI (m)."""
        return np.asarray(self.T.translation(), float)

    # --- manifold ops (31-dim) ---
    def retract(self, xi: np.ndarray) -> "ImuCalib":
        """Manifold retract: apply the 31-dim tangent (6 pose + td + 24 IMU-intrinsic) -> a new ImuCalib."""
        xi = np.asarray(xi, float)
        return ImuCalib(
            self.T.retract(xi[0:6]),
            self.td + xi[6],
            self.imu_in.retract(xi[7:31]))

    def local_coordinates(self, other: "ImuCalib") -> np.ndarray:
        """Manifold log: tangent from self to other (inverse of retract)."""
        return np.concatenate([
            self.T.localCoordinates(other.T),
            [other.td - self.td],
            self.imu_in.local_coordinates(other.imu_in)])

    def copy(self) -> "ImuCalib":
        """Deep copy (independent Pose3 + IMU intrinsics)."""
        return ImuCalib(gtsam.Pose3(self.T.matrix()), self.td, self.imu_in.copy())

    def __repr__(self):
        return f"ImuCalib(td={self.td:.6f}, p_IsinI={self.p_IsinI}, {self.imu_in})"
