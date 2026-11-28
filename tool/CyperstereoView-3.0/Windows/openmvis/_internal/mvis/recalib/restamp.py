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
"""Re-stamp messages onto the base-IMU clock (t' = t + td)."""
from __future__ import annotations


def shifted_time_ns(t_ns: int, td_s: float) -> int:
    """Shift a nanosecond timestamp by td (seconds) -> the re-stamped time on the base-IMU clock."""
    return int(t_ns) + int(round(td_s * 1e9))


def set_header_stamp_ns(msg, t_ns: int) -> None:
    """Set a ROS message's header.stamp to `t_ns` nanoseconds in place (no-op if it has no header.stamp)."""
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is None:
        return
    sec, nsec = divmod(int(t_ns), 1_000_000_000)
    if hasattr(stamp, "nanosec"):       # rosbags ROS2-style time fields
        stamp.sec, stamp.nanosec = sec, nsec
    elif hasattr(stamp, "nsecs"):       # ROS1-style spelling
        stamp.secs, stamp.nsecs = sec, nsec
