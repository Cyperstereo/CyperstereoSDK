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
"""Apply the calibrated inverse IMU model to a sensor_msgs/Imu message."""
from __future__ import annotations
import numpy as np
from mvis.recalib.calib_spec import ImuCalib

def correct_imu_msg(msg, imu: ImuCalib, remove_bias: bool = True, imu_intrinsics: bool = True):
    """Correct one sensor_msgs/Imu in place via the inverse IMU model: remove the intrinsics (Dw/Da/Tg) and, if `remove_bias`, the static bias. With `imu_intrinsics` off only the bias is subtracted. Returns the message."""
    w_m = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
    a_m = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
    b_g = imu.b_g if remove_bias else np.zeros(3)
    b_a = imu.b_a if remove_bias else np.zeros(3)
    if imu_intrinsics:
        w, a = imu.intr.correct(w_m, a_m, b_g, b_a)
    else:
        w, a = w_m - b_g, a_m - b_a
    msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = map(float, w)
    msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = map(float, a)
    return msg
