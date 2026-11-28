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
"""Single source of truth for the rolling-shutter vocabulary.

The two pipelines historically spelled the same concept differently:
- camera-only: ``CameraConfig.shutter_type`` — a string "global" | "rolling" |
  "auto" (RS enabled iff "rolling"; "auto" is treated as global).
- cam-IMU: ``pin_tr`` — a bool (True = global = pin the readout; False = rolling =
  estimate the readout) + ``rolling_mode`` (row convention: 0 top / 1 mid / 2 bottom).

These helpers define the correspondence in ONE place so both representations agree
(`"rolling"` ⟺ `not pin_tr`). No config restructure — both spellings stay; this just
removes the scattered ad-hoc ``== "rolling"`` / ``not pin_tr`` derivations.
"""

from __future__ import annotations

ROLLING = "rolling"
GLOBAL = "global"
AUTO = "auto"

# Default rolling-shutter row convention (matches CamImuCalibConfig.rolling_mode
# and simulation.camera_measurements): 0 = frame stamped at the top row.
DEFAULT_ROLLING_MODE = 0


def is_rolling(shutter_type: str) -> bool:
    """True iff `shutter_type` means rolling shutter. 'auto'/'global'/None → False."""
    return str(shutter_type).strip().lower() == ROLLING


def pin_tr_for(shutter_type: str) -> bool:
    """Cam-IMU `pin_tr` for a camera `shutter_type` (pin the readout unless rolling)."""
    return not is_rolling(shutter_type)


def shutter_type_for(pin_tr: bool) -> str:
    """Camera `shutter_type` string for a cam-IMU `pin_tr` flag."""
    return GLOBAL if pin_tr else ROLLING


def row_offset(row: float, h: int, mode: int) -> float:
    """Dimensionless rolling-shutter row factor ρ: capture time = td + ρ·tr.

    Modes match MVIS / `simulation.camera_measurements._rs_row_time`:
      0: frame stamped at top    -> ρ = row/H
      1: frame stamped at middle -> ρ = row/H − 0.5
      2: frame stamped at bottom -> ρ = (row − H)/H
    """
    if mode == 0:
        return row / h
    if mode == 1:
        return row / h - 0.5
    return (row - h) / h
