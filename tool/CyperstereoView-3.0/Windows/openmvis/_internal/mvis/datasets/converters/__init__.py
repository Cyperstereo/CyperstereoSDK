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
"""Converter registry: full-name pure functions, one module each.

"EuRoC folder format" = the historical mav0/ layout of the EuRoC MAV dataset
(cam0/, imu0/, data.csv) — the de-facto VI-dataset folder convention.
"""
from .archive import extract_archive
from .euroc_folder_to_rosbag import convert_euroc_folder_to_rosbag
from .image_folder_to_rosbag import convert_image_folder_to_rosbag
from .mcap_to_euroc_folder import convert_mcap_to_euroc_folder
from .penncosyvio_to_rosbag import convert_penncosyvio_sequence_to_rosbag
from .rosbag_mono16_to_mono8 import convert_rosbag_mono16_to_mono8
from .rosbag_to_mcap import convert_rosbag_to_mcap
from .tumvie_to_euroc_folder import convert_tumvie_vidata_to_euroc_folder

CONVERTER_REGISTRY = {
    "extract_archive": extract_archive,
    "convert_euroc_folder_to_rosbag": convert_euroc_folder_to_rosbag,
    "convert_image_folder_to_rosbag": convert_image_folder_to_rosbag,
    "convert_mcap_to_euroc_folder": convert_mcap_to_euroc_folder,
    "convert_penncosyvio_sequence_to_rosbag":
        convert_penncosyvio_sequence_to_rosbag,
    "convert_rosbag_mono16_to_mono8": convert_rosbag_mono16_to_mono8,
    "convert_rosbag_to_mcap": convert_rosbag_to_mcap,
    "convert_tumvie_vidata_to_euroc_folder":
        convert_tumvie_vidata_to_euroc_folder,
}
__all__ = ["CONVERTER_REGISTRY", *CONVERTER_REGISTRY.keys()]
