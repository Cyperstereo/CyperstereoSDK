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
"""SqliteDetectionCache — the sqlite detection cache behind the legacy cache interface.

Drop-in for ``mvis.io.detection_cache.DetectionCache``: same constructor, same
``is_valid``/``save``/``load`` surface, same legacy identity semantics (the md5
over ``path|sorted(board_config)`` — deliberately bug-compatible; the honest
identity is ``CaptureSpec``). The file lands beside the legacy one with a
``.mvisdb`` suffix, so both caches can coexist for an A/B bit-identity check.

This is the production detection cache of the pipelines
(`io.detection_pipeline._make_cache`); the interface it mimics is kept so that
code written against `DetectionCache` keeps working.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mvis.mvisdb.bridge import db_from_frame_observations, frame_observations_from_db
from mvis.mvisdb.manager import DatabaseManager
from mvis.mvisdb.mvisdatabase import BoardType

__all__ = ["SqliteDetectionCache"]


def _capture_identity(source_path: str, board_config: dict) -> str:
    """CaptureSpec-grade identity of a cached capture.

    sha256 over {source string, source stat (mtime_ns, size) when the path
    resolves, the FULL config dict}. Replaces the legacy
    md5(path|5 board fields) that let April-vintage caches validate for
    months (8.28 -> 0.65 px once re-detected on one TUM-VI run). The 'cs3:' prefix (cs2 before the image-decoder change)
    guarantees every legacy-keyed cache mismatches -> invalidates -> the
    corpus re-detect fires by construction, never silently.

    The stat is taken on the source part of a ``io.capture_key`` key (the text
    before its first ``|``), so an image or EuRoC directory still folds in its
    stat now that the key also names the topics."""
    stat = None
    try:
        pp = Path(str(source_path).split("|", 1)[0])
        if pp.exists():
            st = pp.stat()
            stat = [st.st_mtime_ns, st.st_size]
    except OSError:
        stat = None
    payload = json.dumps({"src": str(source_path), "stat": stat,
                          "cfg": board_config},
                         sort_keys=True, default=str)
    return "cs3:" + hashlib.sha256(payload.encode()).hexdigest()


class _NullGeom:
    """The shim stores DETECTIONS, not geometry — the board object exists at
    runtime exactly as it does for the JSON cache."""

    def num_points(self) -> int:
        return 0

    def point_3d(self, pid: int):  # pragma: no cover
        raise KeyError(pid)


class SqliteDetectionCache:
    def __init__(self, cache_path: str):
        self._path = Path(cache_path).with_suffix(".mvisdb")
        self._mgr = DatabaseManager()

    # ── legacy surface ─────────────────────────────────────────────────────
    def is_valid(self, source_path: str, board_config: dict) -> bool:
        if not self._path.exists():
            return False
        try:
            info = self._mgr.capture_info(self._path)
            return info["identity_hash"] == _capture_identity(source_path, board_config)
        except Exception:
            return False

    def save(self, all_frames, source_path: str, board_config: dict,
             detection_histories=None, image_sizes=None, imus=None):
        """imus (capture-completeness): optional [{name, topic, t, w, a}]
        finalized IMU streams; declared + measured in the capture core so a
        cam-IMU run's result_imu rows have their FK target."""
        # row is DERIVED (int(v)); every real detector obeys this, and the
        # store does not carry a separate row column — enforce loudly, never
        # round-trip silently wrong.
        for frames in all_frames.values():
            for f in frames:
                for o in f.observations:
                    if o.row != int(o.pixel[1]):
                        raise ValueError(
                            f"observation row {o.row} != int(v)={int(o.pixel[1])} "
                            "— violates the derived-row invariant")
        sizes = dict(image_sizes or {})
        for cid, frames in all_frames.items():
            if cid not in sizes and frames:
                sizes[cid] = tuple(frames[0].image_size)
        db = db_from_frame_observations(
            all_frames, _NullGeom(), name=str(source_path),
            image_sizes=sizes, board_type=BoardType.APRILGRID,
            corners_per_tag=4)
        if imus:
            from mvis.mvisdb.mvisdatabase import (
                ImuInfo, ImuMeasurements, ImuStream,
            )
            for i, im in enumerate(imus):
                db.imus.append(ImuStream(
                    i, ImuInfo(name=im["name"], topic=im.get("topic")),
                    ImuMeasurements(im["t"], im["w"], im["a"])))
        stash = {
            "shim": {
                "board_config": board_config,
                "image_sizes": {str(k): list(v) for k, v in sizes.items()},
                "detection_histories": {
                    str(k): [[[float(s), int(n), float(m)] for s, n, m in h]
                             for h in v]
                    for k, v in (detection_histories or {}).items()},
            }
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)   # legacy does too
        # replace: legacy overwrites; so do we. The lock file stays: removing it would
        # let the next saver lock a new file while another still holds the old one.
        self._mgr.save(db, self._path, attach_run=False, replace=True,
                       identity_hash=_capture_identity(source_path, board_config),
                       spec_json=json.dumps(stash))

    def load(self, source_path: str, board_config: dict):
        if not self.is_valid(source_path, board_config):
            raise ValueError("Cache is invalid or missing")
        info = self._mgr.capture_info(self._path)
        stash = (json.loads(info["spec_json"] or "{}")).get("shim", {})
        db = self._mgr.load(self._path)
        self.last_db = db               # carrier exposure:
                                        # cache hits hand the db out too
        all_frames = {s.camera_id: frame_observations_from_db(db, s.camera_id)
                      for s in db.streams}
        detection_histories = {
            int(k): [[(s, n, m) for s, n, m in h] for h in v]
            for k, v in stash.get("detection_histories", {}).items()}
        image_sizes = {int(k): tuple(v)
                       for k, v in stash.get("image_sizes", {}).items()}
        return all_frames, detection_histories, image_sizes
