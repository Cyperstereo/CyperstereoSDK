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
"""One reading of an IMU sample: a ROS ``sensor_msgs/Imu`` message and a EuRoC ``imuN/data.csv``
row (``t_ns, w xyz, a xyz``; lines starting with ``#`` are comments). Light on purpose (no
image or solver imports), so `mvis inspect` can use it. The whole-bag readers built on it are
`mvis.io.cam_imu_loader._ros1_imu` and `_euroc_imu`."""
from __future__ import annotations

import csv


def ros1_imu_sample(m):
    """``(w, a)``: the gyro and accel triples of an Imu message, as lists."""
    return ([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z],
            [m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])


def euroc_imu_values(row):
    """``(w, a)`` of one EuRoC IMU csv row, as float lists."""
    return [float(x) for x in row[1:4]], [float(x) for x in row[4:7]]


def euroc_imu_rows(path):
    """Yield ``(t_s, w, a)`` for every data row of a EuRoC IMU csv (``t = int(t_ns) * 1e-9``)."""
    for row in csv.reader(open(path)):
        if row and not row[0].startswith("#"):
            w, a = euroc_imu_values(row)
            yield int(row[0]) * 1e-9, w, a
