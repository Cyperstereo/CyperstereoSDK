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
"""ROS1 bag → ROS2/mcap twin via the rosbags conversion API."""
from __future__ import annotations

from pathlib import Path



def convert_rosbag_to_mcap(input_bag: str | Path,
                           output_mcap: str | Path | None = None) -> Path:
    """Emit an mcap twin next to ``input_bag`` (or at ``output_mcap``)."""
    from rosbags.convert import convert  # lazy: optional extra (CI smoke)
    input_bag = Path(input_bag)
    destination = Path(output_mcap) if output_mcap \
        else input_bag.with_suffix(".mcap")
    if destination.exists():
        return destination
    convert(srcs=[input_bag], dst=destination, dst_storage="mcap",
            dst_version=8, compress=None, compress_mode="file",
            default_typestore=None, typestore=None,
            exclude_topics=[], include_topics=[],
            exclude_msgtypes=[], include_msgtypes=[])
    return destination
