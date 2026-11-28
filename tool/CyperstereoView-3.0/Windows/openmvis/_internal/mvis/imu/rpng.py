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
"""RPNG IMU intrinsic model: upper-triangular Dw/Da, free rotation R_atoI."""

from __future__ import annotations

import numpy as np

from mvis.imu.imu_model import (
    ImuModel, RPNG, _ID_UPPER, upper_tri, jac_upper,
)


class RpngImu(ImuModel):
    """RPNG model (MVIS `imu_model` rpng; the pipeline default).

    Dw and Da are upper-triangular (6 params each); the free SO(3) DOF is the
    accel->IMU rotation R_atoI (R_wtoI = I).
    """

    MODEL = RPNG
    NAME = "rpng"
    N_DW = 6
    HAS_R = True

    build_dw = staticmethod(upper_tri)
    build_da = staticmethod(upper_tri)
    dw_param_jacobian = staticmethod(jac_upper)
    da_param_jacobian = staticmethod(jac_upper)

    @staticmethod
    def identity_dw() -> np.ndarray:
        return _ID_UPPER.copy()

    @staticmethod
    def identity_da() -> np.ndarray:
        return _ID_UPPER.copy()
