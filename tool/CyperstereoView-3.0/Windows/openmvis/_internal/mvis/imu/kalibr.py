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
"""Kalibr IMU intrinsic model: lower-triangular Dw/Da, free rotation R_wtoI."""

from __future__ import annotations

import numpy as np

from mvis.imu.imu_model import (
    ImuModel, KALIBR, _ID_LOWER, lower_tri, jac_lower,
)


class KalibrImu(ImuModel):
    """KALIBR model (MVIS `imu_model` kalibr).

    Dw and Da are lower-triangular (6 params each); the free SO(3) DOF is the
    gyro->IMU rotation R_wtoI (R_atoI = I).
    """

    MODEL = KALIBR
    NAME = "kalibr"
    N_DW = 6
    HAS_R = True

    build_dw = staticmethod(lower_tri)
    build_da = staticmethod(lower_tri)
    dw_param_jacobian = staticmethod(jac_lower)
    da_param_jacobian = staticmethod(jac_lower)

    @staticmethod
    def identity_dw() -> np.ndarray:
        return _ID_LOWER.copy()

    @staticmethod
    def identity_da() -> np.ndarray:
        return _ID_LOWER.copy()
