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
"""GiLabs MCAP episode → EuRoC folder format.

Two steps, each its own module (and CLI): ``gilabs_mcap`` decodes the H.265
video + protobuf IMU of a ``recording.mcap`` into the looperrobotics layout,
then ``looper_mav0`` builds the EuRoC ``mav0/`` tree from it. Needs the
``mcap`` extra (``pip install pymvis[mcap]``).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path



def convert_mcap_to_euroc_folder(mcap_path: str | Path,
                                 output_dir: str | Path) -> Path:
    mcap_path, output_dir = Path(mcap_path), Path(output_dir)
    subprocess.run([sys.executable, "-m", "mvis.datasets.converters.gilabs_mcap",
                    "--mcap", str(mcap_path), "--out", str(output_dir)],
                   check=True)
    subprocess.run([sys.executable, "-m", "mvis.datasets.converters.looper_mav0", "--root", str(output_dir)],
                   check=True)
    return output_dir / "mav0"
