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
"""Data I/O: sources and output writers.

`DataSource`, `ImageSource` and `OutputWriter` load on first access (PEP 562), so the light
helpers here (`mvis.io.fileutil`, `mvis.io.imu_reader`) do not pull in numpy, cv2 or yaml."""

_LAZY = {
    "DataSource": "mvis.io.data_source",
    "ImageSource": "mvis.io.image_source",
    "OutputWriter": "mvis.io.output_writer",
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib
        value = getattr(importlib.import_module(_LAZY[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
