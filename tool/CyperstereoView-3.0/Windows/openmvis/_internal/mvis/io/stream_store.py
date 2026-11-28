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
"""Shared per-camera-stream detection store: detect each recording once.

The camera-only pipeline (``DetectionPipeline``) and the cam-IMU loaders
(``io.cam_imu_loader``) each keep their own detection cache, keyed on the whole
capture (every camera topic, the cam-IMU time window, ``min_obs``, ...). Two runs
on one bag therefore never share detections: a camera-only solve, a per-camera
solve and the cam-IMU solve detected the same frames three times (CyberStereo,
4 × 1,120 frames: about 20 min each).

This store keys ONE camera stream instead: the bag, the image topic, header
stamps and the board fingerprint. It keeps every frame with at least one corner
over the time span it covers, so any later run filters by its own window and
corner minimum. Both pipelines decode with ``io.image_decode.ros_image_to_gray``,
stamp with ``io.ros_stamp.header_stamp_s`` and build the board from the same
config, so the detections are the ones either would have made.

Used only when a cache directory is configured (camera-only ``input.cache_dir``,
cam-IMU ``input.cache_dir``); files live in ``<cache_dir>/streams/``.
ROS1 bags only for now.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path

from mvis.core.types import FrameObservations
from mvis.io.capture_key import board_fingerprint, capture_key

logger = logging.getLogger("mvis.io.stream_store")

__all__ = ["StreamDetectionStore", "STREAM_DIR"]

#: Sub-directory of a cache dir that holds the per-stream files.
STREAM_DIR = "streams"
_META_VERSION = 1


class StreamDetectionStore:
    """Per-camera-stream detections under ``<cache_dir>/streams/``."""

    def __init__(self, cache_dir):
        self._dir = Path(cache_dir) / STREAM_DIR

    # ── identity ───────────────────────────────────────────────────────
    @staticmethod
    def _key(source, topic) -> str:
        return capture_key(os.path.abspath(str(source)), [topic], stamp="header")

    def _paths(self, source, topic, board):
        key = self._key(source, topic)
        fp = board_fingerprint(board)
        h = hashlib.md5((key + json.dumps(fp, sort_keys=True, default=str))
                        .encode()).hexdigest()[:10]
        stem = Path(str(source)).name.replace(".", "_")
        tname = str(topic).strip("/").replace("/", "_") or "stream"
        base = self._dir / f"{stem}__{tname}__{h}"
        return key, fp, base, base.with_suffix(".json")

    def _cache(self, base):
        from mvis.io.detection_pipeline import _make_cache
        return _make_cache(str(base))

    # ── read ───────────────────────────────────────────────────────────
    def get(self, source, topic, board, *, camera_id: int = 0,
            lo: float = -math.inf, hi: float = math.inf, min_obs: int = 1):
        """Frames of ``topic`` with ``lo <= t <= hi`` and ``>= min_obs`` corners,
        as ``FrameObservations`` for ``camera_id``; ``None`` if the store has no
        valid entry covering ``[lo, hi]`` (then the caller detects)."""
        key, fp, base, meta_path = self._paths(source, topic, board)
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            return None
        if meta.get("version") != _META_VERSION or meta.get("key") != key:
            return None
        if not (_lo(meta) <= lo and _hi(meta) >= hi):
            return None
        cache = self._cache(base)
        if not cache.is_valid(key, fp):
            return None
        try:
            frames_by_cam, _hist, _sizes = cache.load(key, fp)
        except Exception as e:                       # corrupt / concurrent write
            logger.warning(f"stream store: {base.name} unreadable ({e}); re-detecting")
            return None
        frames = frames_by_cam.get(0, [])
        out = [FrameObservations(camera_id=camera_id, timestamp=f.timestamp,
                                 observations=f.observations, image_size=f.image_size)
               for f in frames
               if lo <= f.timestamp <= hi and len(f.observations) >= min_obs]
        logger.info(f"  stream store hit: {topic} — {len(out)} frames "
                    f"(of {meta.get('frames_read', '?')} read)")
        return out

    def get_all(self, source, topics, board, **kw):
        """``{camera_id: frames}`` for every topic (camera id = position), or
        ``None`` if any topic misses."""
        out = {}
        for cid, topic in enumerate(topics):
            frames = self.get(source, topic, board, camera_id=cid, **kw)
            if frames is None:
                return None
            out[cid] = frames
        return out

    def _covers(self, key, fp, base, meta_path, lo, hi) -> bool:
        """True if a valid stored entry spans ``[lo, hi]``."""
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            return False
        if meta.get("version") != _META_VERSION or meta.get("key") != key:
            return False
        if not (_lo(meta) <= lo and _hi(meta) >= hi):
            return False
        try:
            return bool(self._cache(base).is_valid(key, fp))
        except Exception:
            return False

    # ── write ──────────────────────────────────────────────────────────
    def put(self, source, topic, board, frames, *, frames_read: int,
            lo: float = -math.inf, hi: float = math.inf) -> None:
        """Store every frame of ``topic`` with >= 1 corner, read over
        ``[lo, hi]`` (header seconds; ±inf = the whole recording).

        A valid entry that already covers ``[lo, hi]`` is kept: it holds a
        superset of the same detections, and a narrower cam-IMU window must not
        cut down a full-recording entry."""
        key, fp, base, meta_path = self._paths(source, topic, board)
        if self._covers(key, fp, base, meta_path, lo, hi):
            logger.info(f"  stream store: {topic} already covered — kept")
            return
        kept = [FrameObservations(camera_id=0, timestamp=f.timestamp,
                                  observations=f.observations, image_size=f.image_size)
                for f in frames if len(f.observations) >= 1]
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            meta_path.unlink(missing_ok=True)        # invalid while rewriting
            sizes = {0: tuple(kept[0].image_size)} if kept else {}
            self._cache(base).save({0: kept}, key, fp, {}, sizes)
            meta_path.write_text(json.dumps({
                "version": _META_VERSION, "key": key, "topic": str(topic),
                "lo": _enc(lo), "hi": _enc(hi), "frames_read": int(frames_read),
                "frames_stored": len(kept)}, indent=1))
        except Exception as e:                       # a cache must never fail a run
            logger.warning(f"stream store: could not save {topic} ({e})")
            return
        logger.info(f"  stream store: saved {topic} — {len(kept)}/{frames_read} frames")


def _enc(x: float):
    return None if math.isinf(x) else float(x)


def _lo(meta) -> float:
    v = meta.get("lo")
    return -math.inf if v is None else float(v)


def _hi(meta) -> float:
    v = meta.get("hi")
    return math.inf if v is None else float(v)
