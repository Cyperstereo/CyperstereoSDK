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
"""The sensor-clock sidecar (`*_sensor_clock.json`): per-camera stamp -> capture-time maps. Pure I/O.

Written by `mvis cam-imu` and the pipeline's crash-safe dump; read back by `retime_map` and the bag
restamp tool.
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

FORMAT = "mvis-sensor-clock/1"


def camera_record(rep: dict, topic=None) -> dict:
    """One camera's sidecar entry from its report (`rep` must carry `retime_map`)."""
    return {
        "topic": topic,
        "period_s": rep.get("period_s"), "phase_s": rep.get("phase_s"),
        "rate_hz": rep.get("rate_hz"), "coherence": rep.get("coherence"),
        "scatter_ms": rep.get("scatter_ms"),
        "map": [[float(a), float(b)] for a, b in rep["retime_map"]],
        "dropped_stamps": [float(t) for t in (rep.get("dropped_stamps") or [])]}


def write_sidecar(path, records: dict) -> None:
    """Write {camera key: camera_record} to `path`."""
    with open(path, "w") as f:
        json.dump({"format": FORMAT, "cameras": records}, f)


def read_sidecar(path, check_format: bool = False) -> dict:
    """The sidecar's dict; with `check_format`, ValueError if it is some other JSON."""
    with open(path) as f:
        sc = json.load(f)
    if check_format and sc.get("format", "").split("/")[0] != FORMAT.split("/")[0]:
        raise ValueError(f"{path}: not a sensor-clock sidecar")
    return sc


def write_sensor_clock_sidecar(report: dict, path, cam_topics=None) -> dict:
    """Split the pipeline's sensor-clock report: per-frame stamp maps -> `path`
    (JSON sidecar, reused by `retime_map` and the bag restamp tool); the compact
    per-camera report (rate, coherence, scatter, counts, sidecar path) is returned
    for the summary. `path` None keeps everything compact and drops the maps."""
    compact, maps = {}, {}
    for c, rep in (report or {}).items():
        full, rep = rep, dict(rep)
        m = rep.pop("retime_map", None)
        rep.pop("dropped_stamps", None)
        if m is not None:
            maps[str(int(c))] = camera_record(
                full, cam_topics[int(c)] if cam_topics and int(c) < len(cam_topics) else None)
        compact[str(int(c))] = {k: (round(v, 6) if isinstance(v, float) else v)
                                for k, v in rep.items()}
    if path and maps:
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        write_sidecar(path, maps)
        for c in maps:
            compact[c]["sidecar"] = path
        logger.info("wrote sensor-clock stamp maps %s", path)
    return compact


def load_retime_map(path: str, cam_index: int) -> np.ndarray:
    """(N, 2) [stamp, capture time] for camera `cam_index` from a sensor-clock
    sidecar written by `write_sensor_clock_sidecar`."""
    cams = read_sidecar(path).get("cameras", {})
    key = str(int(cam_index))
    if key not in cams:
        raise ValueError(f"{path}: no retime map for camera {cam_index} "
                         f"(has {sorted(cams)})")
    m = np.asarray(cams[key]["map"], float).reshape(-1, 2)
    logger.info("camera %d: retime map %s (%d frames, %.4f Hz)", cam_index, path,
                len(m), float(cams[key].get("rate_hz") or 0.0))
    return m
