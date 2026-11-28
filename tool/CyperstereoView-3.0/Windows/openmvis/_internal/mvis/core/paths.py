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
"""Where the MVIS calibration datasets live.

`datasets_root()` is ``$MVIS_DATASETS`` when set (used as written), else ``~/Datasets``
expanded. `DATASETS_DEFAULT` is the unexpanded default that ``${MVIS_DATASETS}`` in a config
falls back to (`config_parser.expand_path_vars`).
"""
from __future__ import annotations

import os

DATASETS_ENV = "MVIS_DATASETS"
DATASETS_DEFAULT = "~/Datasets"


def datasets_root() -> str:
    """``$MVIS_DATASETS`` if set, else the expanded ``~/Datasets``. Read at call time."""
    return os.environ.get(DATASETS_ENV, os.path.expanduser(DATASETS_DEFAULT))
