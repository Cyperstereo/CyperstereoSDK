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
"""Detection cache: save and load board detections in JSON format.

JSON is human-readable, inspectable, and portable across Python versions.
"""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from mvis.core.types import FrameObservations, Observation

logger = logging.getLogger(__name__)


def _compute_cache_key(source_path: str, board_config: dict) -> str:
    key_str = f"{source_path}|{sorted(board_config.items())}"
    return hashlib.md5(key_str.encode()).hexdigest()[:12]


def legacy_json_frames(cache_path) -> dict:
    """Read-only: ``{camera_id: [FrameObservations]}`` from a legacy ``detections.json``
    without checking its cache key. A missing file gives {}; an unreadable one logs a
    warning and gives {}. Absent ``row`` / ``image_size`` fields default to 0 / (0, 0)."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path) as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"  Failed to parse {cache_path}: {e}")
        return {}
    all_frames = {}
    for cam_id_str, cam_data in data.get("cameras", {}).items():
        cam_id = int(cam_id_str)
        frames = []
        for fd in cam_data.get("frames", []):
            observations = [
                Observation(
                    point_id=o["point_id"],
                    pixel=np.array(o["pixel"]),
                    row=o.get("row", 0),
                )
                for o in fd.get("observations", [])
            ]
            frames.append(FrameObservations(
                camera_id=fd["camera_id"],
                timestamp=fd["timestamp"],
                observations=observations,
                image_size=tuple(fd.get("image_size", [0, 0])),
            ))
        all_frames[cam_id] = frames
    return all_frames


class DetectionCache:
    """Save and load board detections in JSON format: the cache format before the
    sqlite store. The pipelines write `.mvisdb` captures now (`mvis.mvisdb.shim.
    SqliteDetectionCache`, same `is_valid`/`save`/`load` surface) and read old
    `detections.json` files through `legacy_json_frames`; this class remains for
    code that wants a plain-JSON copy of a frame set."""

    def __init__(self, cache_path: str):
        self._path = Path(cache_path).with_suffix(".json")

    def is_valid(self, source_path: str, board_config: dict) -> bool:
        """True if a cached detection set exists for this source path + board config (path present and the stored digest matches)."""
        if not self._path.exists():
            return False
        try:
            with open(self._path) as f:
                data = json.load(f)
            return data.get("cache_key") == _compute_cache_key(source_path, board_config)
        except Exception:
            return False

    def save(
        self,
        all_frames: dict[int, list[FrameObservations]],
        source_path: str,
        board_config: dict,
        detection_histories=None,
        image_sizes=None,
        imus=None,          # accepted for API parity; JSON cache stores no IMU
    ):
        """Write the detected frames to the on-disk cache, keyed by source path + board config."""
        self._path.parent.mkdir(parents=True, exist_ok=True)

        cameras = {}
        for cam_id, frames in all_frames.items():
            cam_frames = []
            for frame in frames:
                obs_list = [
                    {"point_id": int(o.point_id),
                     "pixel": [float(o.pixel[0]), float(o.pixel[1])],
                     "row": int(o.row)}
                    for o in frame.observations
                ]
                cam_frames.append({
                    "timestamp": float(frame.timestamp),
                    "image_size": list(frame.image_size),
                    "camera_id": int(frame.camera_id),
                    "num_observations": len(obs_list),
                    "observations": obs_list,
                })
            cameras[str(cam_id)] = {"frames": cam_frames}

        # Convert detection histories
        hist_out = {}
        if detection_histories:
            for cam_id, hists in detection_histories.items():
                hist_out[str(cam_id)] = [
                    [[float(s), int(n), float(m)] for s, n, m in h]
                    for h in hists
                ]

        img_sizes_out = {}
        if image_sizes:
            for cam_id, sz in image_sizes.items():
                img_sizes_out[str(cam_id)] = list(sz)

        data = {
            "cache_key": _compute_cache_key(source_path, board_config),
            "source_path": source_path,
            "board_config": board_config,
            "image_sizes": img_sizes_out,
            "detection_histories": hist_out,
            "cameras": cameras,
        }

        with open(self._path, "w") as f:
            json.dump(data, f, indent=1)

        total = sum(len(frames) for frames in all_frames.values())
        size_mb = self._path.stat().st_size / 1024 / 1024
        logger.info(f"Detection cache saved: {self._path} "
                     f"({total} frames, {size_mb:.1f} MB)")

    def load(
        self,
        source_path: str,
        board_config: dict,
    ) -> tuple[dict[int, list[FrameObservations]], dict, dict]:
        """Load cached detections for `source_path` when the cache is valid; returns the frames, or None on a miss."""
        if not self.is_valid(source_path, board_config):
            raise ValueError("Cache is invalid or missing")

        with open(self._path) as f:
            data = json.load(f)

        all_frames = {}
        for cam_id_str, cam_data in data["cameras"].items():
            cam_id = int(cam_id_str)
            frames = []
            for fd in cam_data["frames"]:
                observations = [
                    Observation(
                        point_id=o["point_id"],
                        pixel=np.array(o["pixel"]),
                        row=o["row"],
                    )
                    for o in fd["observations"]
                ]
                frames.append(FrameObservations(
                    camera_id=fd["camera_id"],
                    timestamp=fd["timestamp"],
                    observations=observations,
                    image_size=tuple(fd["image_size"]),
                ))
            all_frames[cam_id] = frames

        # Load detection histories
        detection_histories = {}
        for cam_id_str, hists in data.get("detection_histories", {}).items():
            detection_histories[int(cam_id_str)] = [
                [(s, n, m) for s, n, m in h] for h in hists
            ]

        # Load image sizes
        image_sizes = {}
        for cam_id_str, sz in data.get("image_sizes", {}).items():
            image_sizes[int(cam_id_str)] = tuple(sz)

        total = sum(len(f) for f in all_frames.values())
        logger.info(f"Detection cache loaded: {self._path} "
                     f"({total} frames, {len(all_frames)} cameras)")

        return all_frames, detection_histories, image_sizes
