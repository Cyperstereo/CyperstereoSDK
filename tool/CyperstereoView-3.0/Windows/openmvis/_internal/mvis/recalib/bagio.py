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
"""Small ROS1 bag helpers shared by the bag rewriters (recalib tools and studies). No rosbags import:
they work on the reader, writer and message objects passed in.

Stamps are integer nanoseconds. The raw-byte helpers read and patch the header stamp of a
serialized ROS1 message whose first field is a ``std_msgs/Header`` (``seq`` uint32, then
``stamp.secs`` and ``stamp.nsecs`` uint32 at bytes 4-12) without deserialising it; shifting in
integer ns keeps every unshifted digit, where a float-seconds round trip near 1.7e9 s rounds
the stamp to about 0.24 us.
"""
from __future__ import annotations

import struct

NS_PER_S = 1_000_000_000
_STAMP = struct.Struct("<II")
_STAMP_OFFSET = 4                  # after the header's uint32 seq


def header_ns(msg) -> int:
    """Message header stamp in nanoseconds (ROS1 `secs/nsecs` or ROS2 `sec/nanosec` field names)."""
    s = msg.header.stamp
    sec = int(getattr(s, "sec", getattr(s, "secs", 0)))
    nsec = int(getattr(s, "nanosec", getattr(s, "nsecs", 0)))
    return sec * 1_000_000_000 + nsec


def copy_connections(reader, writer) -> dict:
    """Recreate every connection on the writer (preserves msgdef + md5sum): {reader id: writer connection}."""
    return {c.id: writer.add_connection(c.topic, c.msgtype, msgdef=c.msgdef.data,
                                        md5sum=c.digest, callerid=getattr(c.ext, 'callerid', None),
                                        latching=getattr(c.ext, 'latching', None))
            for c in reader.connections}


def to_sec_nsec(t: float) -> tuple[int, int]:
    """Float seconds -> (sec, nsec) with nsec in [0, 1e9)."""
    sec = int(t)
    nsec = int(round((t - sec) * 1e9))
    if nsec >= 10**9:
        sec += 1
        nsec -= 10**9
    if nsec < 0:
        sec -= 1
        nsec += 10**9
    return sec, nsec


def ns_to_sec_nsec(ns: int) -> tuple[int, int]:
    """Integer ns -> (sec, nsec) with nsec in [0, 1e9)."""
    return divmod(int(ns), NS_PER_S)


def seconds_to_ns(t: float) -> int:
    """A duration or offset in float seconds -> the nearest integer ns."""
    return int(round(float(t) * NS_PER_S))


def raw_stamp_ns(raw) -> int:
    """Header stamp (ns) of a serialized ROS1 message that starts with a std_msgs/Header."""
    sec, nsec = _STAMP.unpack_from(raw, _STAMP_OFFSET)
    return sec * NS_PER_S + nsec


def with_raw_stamp_ns(raw, ns: int) -> bytes:
    """``raw`` with its header stamp set to ``ns``; every other byte unchanged."""
    return (bytes(raw[:_STAMP_OFFSET]) + _STAMP.pack(*ns_to_sec_nsec(ns))
            + bytes(raw[_STAMP_OFFSET + _STAMP.size:]))
