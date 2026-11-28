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
"""IMU intrinsic models (Kalibr / RPNG / Calyx) — value-side 24-dim manifold,
correction matrices, manifold ops, and the forward/inverse measurement model."""

from mvis.imu.imu_model import (
    ImuModel, create_imu_model,
    KALIBR, RPNG, CALYX, IMU_STAGES,
    parse_imu_model, imu_model_name, n_dw_for,
    dw_matrix, da_matrix, tg_matrix,
    dw_jacobian, da_jacobian, tg_jacobian, decompose_dw,
    intrinsic_block_slices, freed_intrinsic_blocks, intrinsic_sigma_vector,
)
from mvis.imu.kalibr import KalibrImu
from mvis.imu.rpng import RpngImu
from mvis.imu.calyx import CalyxImu
