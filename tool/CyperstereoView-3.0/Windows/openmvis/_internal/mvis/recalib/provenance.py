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
# src/mvis/recalib/provenance.py
"""Audit sidecar describing how a corrected bag was produced."""
from __future__ import annotations
import os, yaml

from mvis.io.fileutil import compute_sha256

SCHEMA_VERSION = 1

def _sha256(path: str):
    if not os.path.exists(path):
        return None
    return compute_sha256(path)

def write_provenance(out_bag: str, *, source_bag, calib_path, mapping,
                     remove_bias, undistort, base_imu_topic,
                     corrected_topics, passthrough_topics,
                     imu_intrinsics=True, td=True, t_ga=True, extra=None) -> str:
    """Write a `<out_bag>.recalib.yaml` provenance sidecar: source bag, calib path, mapping, flags, and which topics were corrected vs passed through; `extra` adds or overrides keys. Returns the sidecar path."""
    abs_calib = os.path.abspath(calib_path)
    doc = {
        "schema_version": SCHEMA_VERSION,
        "source_bag": os.path.abspath(source_bag),
        "calib_path": abs_calib,
        "calib_sha256": _sha256(abs_calib),
        "timeline": "base_imu_clock" if td else "original timestamps",
        "base_imu_topic": base_imu_topic,
        "td_convention": "t_on_base = t_sensor + td",
        "corrections": {"remove_bias": bool(remove_bias),
                        "undistort": bool(undistort),
                        "imu_intrinsics": bool(imu_intrinsics),
                        "td": bool(td), "t_ga": bool(t_ga)},
        "rolling_shutter": "frame-stamp only; per-row unwarp NOT applied",
        "topic_mapping": dict(mapping),
        "corrected_topics": list(corrected_topics),
        "passthrough_topics": list(passthrough_topics),
    }
    doc.update(extra or {})
    path = out_bag + ".recalib.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    return path
