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
"""Observation-level outlier rejection for a calibration pass.

Given a `CalibrationResult` and the original `all_frames`, recompute the
residual of every observation using the shared projection model and filter
out the worst ones so a second calibration pass can refine the model on a
cleaner dataset.

Design decisions:

- **Observation-level, not frame-level**: a frame may contain mostly good
  tag corners and a few bad ones; we keep the good ones and only drop the
  frame if too few observations remain to solve PnP.
- **Threshold = max(absolute_px, percentile)**: the absolute minimum
  prevents over-pruning an already-clean calibration (e.g. don't prune
  observations at 0.7 px even if they are at the 99.5 percentile); the
  percentile catches catastrophic outliers with huge residuals.
- **Per-camera thresholds**: the percentile is computed per-camera so a
  high-distortion fisheye's wider spread doesn't drag down a well-behaved
  pinhole.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from mvis.detection.board import Board
# board_pose_timeline lived here before ResultProjector; kept importable from this module
from mvis.calibration.result_projector import ResultProjector, board_pose_timeline  # noqa: F401
from mvis.core.types import (
    CalibrationResult,
    FrameObservations,
    Observation,
)

logger = logging.getLogger(__name__)


@dataclass
class OutlierFilterStats:
    """Summary of one pass of outlier filtering."""

    camera_id: int
    threshold_px: float
    n_observations_before: int
    n_observations_after: int
    n_frames_before: int
    n_frames_after: int


def compute_per_observation_residuals(
    result: CalibrationResult,
    all_frames: dict[int, list[FrameObservations]],
    board: Board,
    image_sizes: dict[int, tuple[int, int]],
) -> dict[int, list[list[Optional[float]]]]:
    """Return residual[cam_id][frame_idx] = list of per-observation residuals.

    Projects through `ResultProjector`, so residuals match the optimizer's cost and score each
    corner against the board the solve fitted (refined when it estimated the board shape).
    A residual is `None` if projection fails for that observation (cheirality,
    or frame outside the ref-camera timeline).
    """
    proj = ResultProjector(result, board, image_sizes)
    out: dict[int, list[list[Optional[float]]]] = {}
    for cam_id, frames in all_frames.items():
        if not proj.has_camera(cam_id):
            continue
        cam = proj.camera(cam_id)
        per_frame_residuals: list[list[Optional[float]]] = []
        for frame in frames:
            b = proj.bracket(frame.timestamp)
            if b is None:
                per_frame_residuals.append([None] * len(frame.observations))
                continue
            per_frame_residuals.append(
                [proj.residual(cam, obs, b) for obs in frame.observations])
        out[cam_id] = per_frame_residuals
    return out


def filter_outliers(
    all_frames: dict[int, list[FrameObservations]],
    residuals: dict[int, list[list[Optional[float]]]],
    board: Board,
    *,
    absolute_threshold_px: float = 2.0,
    percentile: float = 99.0,
    frame_median_threshold_px: float | None = None,
    frame_median_percentile: float = 99.0,
) -> tuple[dict[int, list[FrameObservations]], dict[int, OutlierFilterStats]]:
    """Drop outlier observations AND outlier frames.

    Two-stage filtering:

    1. **Frame-level:** drop entire frames whose *median* residual is far
       above the rest. A whole bad frame (wrong pose, heavy motion blur)
       can contaminate the calibration even if individual observations
       look OK. Threshold =
       max(frame_median_threshold_px, percentile of per-frame medians).

    2. **Observation-level:** in the surviving frames, drop individual
       observations whose residual exceeds the per-camera pixel threshold.
       Threshold = max(absolute_threshold_px, percentile of all residuals).

    Observations whose projection failed (residual is `None`) are always
    dropped. If a frame falls below `board.min_points_for_pose()` after
    observation-level filtering, the whole frame is removed too.

    Args:
        all_frames: camera_id -> list of FrameObservations.
        residuals: camera_id -> list (per frame) of list (per obs) of
            residual magnitudes. `None` means projection failed.
        board: Board object (for `min_points_for_pose`).
        absolute_threshold_px: lower bound for the obs-level threshold.
        percentile: percentile of all per-obs residuals for obs-level
            threshold.
        frame_median_threshold_px: lower bound for the frame-level
            threshold. If None, defaults to 2× `absolute_threshold_px`.
        frame_median_percentile: percentile of per-frame medians for the
            frame-level threshold.

    Returns:
        (filtered_all_frames, per-camera stats)
    """
    min_pts = board.min_points_for_pose()
    out_frames: dict[int, list[FrameObservations]] = {}
    stats: dict[int, OutlierFilterStats] = {}

    if frame_median_threshold_px is None:
        frame_median_threshold_px = 2.0 * absolute_threshold_px

    for cam_id, frames in all_frames.items():
        n_obs_before = sum(len(f.observations) for f in frames)
        n_frames_before = len(frames)
        frame_res_list = residuals.get(cam_id, [])

        # ----- Stage A: per-frame median threshold -----
        frame_medians: list[float] = []
        for fi, frame in enumerate(frames):
            per_obs = frame_res_list[fi] if fi < len(frame_res_list) else []
            valid = [r for r in per_obs if r is not None]
            frame_medians.append(float(np.median(valid)) if valid else float("inf"))

        finite_medians = [m for m in frame_medians if np.isfinite(m)]
        if finite_medians:
            pct_frame = float(np.percentile(finite_medians, frame_median_percentile))
            frame_threshold = max(frame_median_threshold_px, pct_frame)
        else:
            frame_threshold = frame_median_threshold_px

        n_frame_rejected = sum(1 for m in frame_medians if m > frame_threshold)

        # ----- Stage B: per-camera per-obs threshold (on surviving frames) -----
        surviving_obs = [r
                         for fi, per_obs in enumerate(frame_res_list)
                         if fi < len(frame_medians) and frame_medians[fi] <= frame_threshold
                         for r in per_obs if r is not None]
        if surviving_obs:
            pct_obs = float(np.percentile(surviving_obs, percentile))
            obs_threshold = max(absolute_threshold_px, pct_obs)
        else:
            obs_threshold = absolute_threshold_px

        # ----- Apply both filters -----
        new_frames: list[FrameObservations] = []
        for fi, frame in enumerate(frames):
            if frame_medians[fi] > frame_threshold:
                continue  # whole frame dropped
            per_obs = frame_res_list[fi] if fi < len(frame_res_list) else []
            kept_obs: list[Observation] = []
            for oi, obs in enumerate(frame.observations):
                r = per_obs[oi] if oi < len(per_obs) else None
                if r is None or r > obs_threshold:
                    continue
                kept_obs.append(obs)
            if len(kept_obs) < min_pts:
                continue
            new_frames.append(FrameObservations(
                camera_id=frame.camera_id,
                timestamp=frame.timestamp,
                observations=kept_obs,
                image_size=frame.image_size,
            ))

        n_obs_after = sum(len(f.observations) for f in new_frames)
        n_frames_after = len(new_frames)
        out_frames[cam_id] = new_frames
        # Report the obs-level threshold in the stats (the one users tune).
        stats[cam_id] = OutlierFilterStats(
            camera_id=cam_id,
            threshold_px=obs_threshold,
            n_observations_before=n_obs_before,
            n_observations_after=n_obs_after,
            n_frames_before=n_frames_before,
            n_frames_after=n_frames_after,
        )
        logger.info(
            f"Camera {cam_id}: frame-median threshold={frame_threshold:.2f} px "
            f"(rejected {n_frame_rejected} frames), "
            f"obs threshold={obs_threshold:.2f} px — "
            f"{n_obs_before} → {n_obs_after} obs, "
            f"{n_frames_before} → {n_frames_after} frames"
        )
    return out_frames, stats
