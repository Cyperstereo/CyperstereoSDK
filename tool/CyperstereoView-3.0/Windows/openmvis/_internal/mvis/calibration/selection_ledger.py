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
"""SelectionLedger: every permanent frame removal, recorded at the moment of
surgery.

The camera-only pipeline has 14 mutation sites expressed as list identity
destruction. Before any of them can become db flags, the flags must be
TRUTHFUL — so this ledger shadows the surgery first: each site records
(camera, timestamp, stage:reason) as it drops, and the run-attach step in
scripts/run_calibration.py folds the ledger into MVISDatabase frame flags
(is_selected_to_use=False + exclude_reason) before save. Solver inputs are
untouched; the overlay simply stops lying about what the solve used.

Step-1 scope: whole-frame removals from the joint solve stream only.
Stage-local subsets (init-frame selection, stage-2 hz thinning) are not
exclusions and are NOT recorded; observation-level drops (corner subsample,
outlier corners) come in a later step.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def mark_frames(db, kept_by_cam: dict[int, set[float]],
                stage: str, reason: str) -> int:
    """Direct flag write (2c-4): mark every still-selected db frame whose
    timestamp is not in its camera's keep-set. First-exclusion-wins like the
    ledger — an upstream reason is never overwritten. Cameras absent from
    kept_by_cam are left untouched (unknown, not dropped). Returns count."""
    n = 0
    for stream in db.streams:
        if stream.camera_id not in kept_by_cam:
            continue
        kept = kept_by_cam[stream.camera_id]
        for f in stream.frames:
            if f.is_selected_to_use and f.timestamp not in kept:
                f.is_selected_to_use = False
                f.exclude_reason = f"{stage}:{reason}"
                n += 1
    return n


class SelectionLedger:
    """First drop wins: a frame already dropped at an earlier stage keeps its
    earlier reason (later stages never saw it)."""

    def __init__(self, db=None) -> None:
        self.drops: dict[tuple[int, float], str] = {}
        self.drops_obs: dict[tuple[int, float, int], tuple[str, float | None]] = {}
        self._db = db                   # write-through target (2c-5)
        self._frame_index = None        # lazy {(cam, ts): FrameData}
        self._obs_index = None          # lazy {(cam, ts, pid): TagDetection}

    def _frame(self, cam_id: int, timestamp: float):
        if self._db is None:
            return None
        if self._frame_index is None:
            self._frame_index = {(s.camera_id, f.timestamp): f
                                 for s in self._db.streams for f in s.frames}
        return self._frame_index.get((cam_id, timestamp))

    def record(self, cam_id: int, timestamp: float, stage: str, reason: str) -> None:
        key = (cam_id, float(timestamp))
        if key in self.drops:
            return
        self.drops[key] = f"{stage}:{reason}"
        f = self._frame(cam_id, float(timestamp))
        if f is not None and f.is_selected_to_use:  # upstream reason wins
            f.is_selected_to_use = False
            f.exclude_reason = f"{stage}:{reason}"

    def record_obs(self, cam_id: int, timestamp: float, point_id: int,
                   stage: str, reason: str,
                   residual_px: float | None = None,
                   residual_source=None) -> None:
        """Observation-level drop (validity is residual-based — the
        write-through sets is_valid=False and stores the Residual that
        condemned the corner). Index key is (cam, ts, point_id): with
        multiple boards per frame the first matching detection wins."""
        key = (cam_id, float(timestamp), int(point_id))
        if key in self.drops_obs:
            return
        self.drops_obs[key] = (f"{stage}:{reason}",
                               None if residual_px is None else float(residual_px))
        if self._db is None:
            return
        if self._obs_index is None:
            self._obs_index = {}
            for s in self._db.streams:
                for f in s.frames:
                    for g in f.board_detections:
                        for d in g.detections:
                            self._obs_index.setdefault(
                                (s.camera_id, f.timestamp, d.point_id), d)
        d = self._obs_index.get(key)
        if d is not None and d.is_valid:
            d.is_valid = False
            d.exclude_reason = f"{stage}:{reason}"
            if residual_px is not None:
                from mvis.mvisdb.mvisdatabase import PoseSource, Residual
                d.residual = Residual(float(residual_px),
                                      residual_source or PoseSource.REFINED_CAM_ONLY)

    def unmark(self, stage_prefix: str) -> int:
        """Revert every mark a stage wrote. First-drop-wins
        means a frame/observation this stage excluded had NO earlier
        exclusion, so reverting restores it to selected/valid outright.
        Returns the number of entries reverted."""
        n = 0
        for key in [k for k, v in self.drops.items()
                    if v.startswith(stage_prefix)]:
            del self.drops[key]
            f = self._frame(*key)
            if f is not None:
                f.is_selected_to_use = True
                f.exclude_reason = None
            n += 1
        for key in [k for k, (v, _) in self.drops_obs.items()
                    if v.startswith(stage_prefix)]:
            del self.drops_obs[key]
            d = (self._obs_index or {}).get(key)
            if d is not None:
                d.is_valid = True
                d.exclude_reason = None
                d.residual = None
            n += 1
        return n

    def diff(self, before: dict[int, list], after: dict[int, list],
             stage: str, reason: str) -> int:
        """Record every frame present in `before` but absent in `after`
        (matched by exact timestamp — timestamps are carried verbatim through
        the pipeline, never recomputed). Returns the number recorded."""
        n = 0
        for cid, frames in before.items():
            kept = {f.timestamp for f in after.get(cid, [])}
            for f in frames:
                if f.timestamp not in kept:
                    self.record(cid, f.timestamp, stage, reason)
                    n += 1
        return n

    def diff_one(self, cam_id: int, before: list, after: list,
                 stage: str, reason: str) -> int:
        return self.diff({cam_id: before}, {cam_id: after}, stage, reason)

    def apply_to_db(self, db, solve_frames: dict[int, list] | None = None) -> int:
        """Fold the ledger into MVISDatabase inline flags. When `solve_frames`
        (the full-rate detection dict the solve started from) is given, db
        frames absent from it are additionally marked as detection-stage
        thinning — the drop that happened before this ledger existed.
        Returns the number of frames flagged."""
        n = 0
        for stream in db.streams:
            started = ({f.timestamp for f in solve_frames.get(stream.camera_id, [])}
                       if solve_frames is not None else None)
            for f in stream.frames:
                key = (stream.camera_id, f.timestamp)
                if key in self.drops and f.is_selected_to_use:
                    # idempotent over write-through (2c-5)
                    f.is_selected_to_use = False
                    f.exclude_reason = self.drops[key]
                    n += 1
                elif (started is not None and f.timestamp not in started
                      and f.is_selected_to_use):
                    # not already excluded upstream (a carried db may hold the
                    # more precise detection:diversity reason -- keep it)
                    f.is_selected_to_use = False
                    f.exclude_reason = "detection:hz_subsample"
                    n += 1
        return n
