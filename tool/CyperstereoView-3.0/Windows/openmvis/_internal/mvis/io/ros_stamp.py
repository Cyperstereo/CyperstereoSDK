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
"""ROS1 message time from the serialized header, without deserialising the message.

Every stamped ROS1 message (Image, CompressedImage, Imu) starts with its std_msgs/Header:
seq (u32), stamp.sec (u32), stamp.nsec (u32). The camera-only readers and the cam-IMU
loader read the time from these bytes, so an out-of-window image is skipped before its
pixels are decoded, and a message whose stamp was never set (0) falls back to the bag's
record time instead of landing at t = 0.
"""
import struct

__all__ = ["header_stamp_s"]


def header_stamp_s(rawdata: bytes, record_ns: int) -> float:
    """Message time from header.stamp, read without decoding the message.

    A serialized ROS1 Image or CompressedImage starts with the header: seq (u32), sec (u32), nsec (u32).
    The bag's record time is only when the message reached the recorder, so it is the
    fallback for a message whose stamp was never set."""
    sec, nsec = struct.unpack_from("<II", rawdata, 4)
    if sec == 0 and nsec == 0:
        return record_ns / 1e9
    return sec + nsec * 1e-9
