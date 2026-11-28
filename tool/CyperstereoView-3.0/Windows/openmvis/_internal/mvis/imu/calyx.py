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
"""Calyx IMU intrinsic model: full 3x3 Dw, upper-tri Da, NO free rotation.

The gyro->IMU rotation is absorbed into the full (9-param) Dw during the solve
and recovered post-hoc by RQ decomposition (`decompose` -> R_wtoI · U_w), so the
model carries no separate SO(3) DOF (R_wtoI = R_atoI = I while optimizing).
"""

from __future__ import annotations

import numpy as np

from mvis.imu.imu_model import (
    ImuModel, CALYX, _ID_UPPER, _DEFAULT_DW_FULL,
    full3x3, upper_tri, jac_full, jac_upper,
)


class CalyxImu(ImuModel):
    """CALYX model (accel-anchored, full-Dw)."""

    MODEL = CALYX
    NAME = "calyx"
    N_DW = 9
    HAS_R = False

    build_dw = staticmethod(full3x3)        # full 3x3 (9 params, row-major)
    build_da = staticmethod(upper_tri)      # accel correction stays upper-tri
    dw_param_jacobian = staticmethod(jac_full)
    da_param_jacobian = staticmethod(jac_upper)

    @staticmethod
    def identity_dw() -> np.ndarray:
        return _DEFAULT_DW_FULL.copy()

    @staticmethod
    def identity_da() -> np.ndarray:
        return _ID_UPPER.copy()
