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
"""One identity for a detection capture, shared by every detection cache.

``capture_key`` names WHICH frames were detected: the source, the camera topics (or image
dirs) in camera order, the time window and any reader knob that selects frames.
``board_fingerprint`` names HOW they were detected: the board's full scalar config. The
camera-only ``DetectionPipeline`` (``scripts/run_calibration.py``, ``mvis cam``) and the
cam-IMU loaders all key their cache on these two, so two runs that read different topics
or windows of one bag can no longer share (or overwrite) one cache entry.
"""
from __future__ import annotations

__all__ = ["board_fingerprint", "board_signature", "capture_key"]


def _scalars(board) -> list[tuple[str, object]]:
    return sorted((k, v) for k, v in vars(board).items()
                  if isinstance(v, (int, float, str, bool)))


def board_fingerprint(board) -> dict:
    """The board's scalar params (spacing, start id, detector knobs, ...) plus its class
    name as ``board_type``: a detector-parameter change gives a different fingerprint."""
    cfg = dict(_scalars(board))
    cfg["board_type"] = type(board).__name__
    return cfg


def board_signature(board) -> str:
    """The same fingerprint as one string, ``ClassName([(key, value), ...])``."""
    return f"{type(board).__name__}({_scalars(board)})"


def capture_key(source, topics=(), window=None, **knobs) -> str:
    """``source|topics=[...]|window=...|knob=value...`` (knobs sorted by name; a None
    window or knob is left out). `topics` keeps camera order: camera ids follow it."""
    parts = [str(source), f"topics={[str(t) for t in topics]}"]
    if window is not None:
        parts.append(f"window={window}")
    parts += [f"{k}={knobs[k]}" for k in sorted(knobs) if knobs[k] is not None]
    return "|".join(parts)

