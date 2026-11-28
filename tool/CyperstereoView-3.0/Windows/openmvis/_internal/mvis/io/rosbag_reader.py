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
"""One door to a ROS bag, ROS1 or ROS2, through the ``rosbags`` library (no ROS install).

A ROS1 bag is a ``.bag`` file. A ROS2 bag is its folder (the one with ``metadata.yaml``) or a
single ``.db3`` / ``.mcap`` file. Both readers give ``connections`` and
``messages(connections=...) -> (connection, record_ns, raw)``, so the callers are the same.

The header stamp is read from the raw bytes by `io.ros_stamp.header_stamp_s` for both: a
little-endian CDR message starts with 4 encapsulation bytes where ROS1 has ``seq``, then
``sec`` and ``nanosec`` at the same offsets. shortcut: big-endian CDR is refused here, not read.
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["is_ros2_bag", "open_bag"]

_ROS2_FILES = (".db3", ".mcap")


def is_ros2_bag(path) -> bool:
    p = Path(path)
    return p.is_dir() or p.suffix in _ROS2_FILES


def open_bag(path):
    """``(reader, deserialize)`` for the bag at ``path``. ``reader`` is the rosbags Reader, to
    use as a context manager; ``deserialize(raw, msgtype)`` gives the message."""
    from rosbags.typesys import Stores, get_typestore

    if not is_ros2_bag(path):
        from rosbags.rosbag1 import Reader
        return Reader(path), get_typestore(Stores.ROS1_NOETIC).deserialize_ros1

    from rosbags.rosbag2 import Reader
    cdr = get_typestore(Stores.ROS2_HUMBLE).deserialize_cdr

    def deserialize(raw, msgtype):
        if bytes(raw[:2]) != b"\x00\x01":       # CDR_LE; header_stamp_s reads little-endian
            raise ValueError(f"{path}: {msgtype} is not little-endian CDR (header {bytes(raw[:2])!r})")
        return cdr(raw, msgtype)

    return Reader(path), deserialize
