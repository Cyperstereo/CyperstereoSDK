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
"""Unified quality gate for frame and observation filtering.

Single entry point for the ``filter → reinit PnP → pass downstream``
pattern used at three places in the calibration pipeline:

- **Stage 1→2**: detection-quality gating (pooled-pose, corner views, min tags)
- **Stage 3**: trajectory alignment (min tags + PnP reprojection gate)
- **Stage 5**: outlier-driven refinement (min tags + PnP reprojection gate)

Each gate is independently toggleable via ``QualityGateConfig``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from mvis.detection.board import Board
from mvis.detection.frame_quality import (
    PoseFilterStats,
    CornerFilterStats,
    _tag_ids,
    filter_ref_frames_by_pooled_tags,
    filter_observations_by_min_views,
    filter_observations_by_min_views_per_camera,
    filter_frames_by_min_tags,
    log_tag_coverage,
)
from mvis.calibration.initializer import Initializer
from mvis.core.types import FrameObservations

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class QualityGateConfig:
    """What to run — each gate is independently toggleable."""

    # Frame-level dual filter (both must pass):
    #   min_tag_count: spatial spread (>= N tags from different board regions)
    #   min_point_count: data quantity (>= N observation points)
    # Legacy: min_tags_per_frame is used as min_tag_count if the new
    # fields are not set (backward compat with existing configs).
    min_tags_per_frame: int = 0   # LEGACY — use min_tag_count instead
    min_tag_count: int = 0        # 0 = use min_tags_per_frame
    min_point_count: int = 0      # 0 = no point-count check
    # Pooled-pose ref-frame filter
    enable_pooled_pose_filter: bool = False
    min_pose_tag_fraction: float = 0.20
    min_pose_tag_floor: int = 4
    pose_sync_tolerance_s: float | None = None
    # Observation-level: global corner views (0 = skip)
    min_corner_views: int = 0
    # Observation-level: per-camera corner views (0 = skip)
    min_corner_views_per_camera: int = 0
    # Tag spread: minimum convex hull of tag centers / image area (0 = skip)
    min_hull_fraction: float = 0.0
    # PnP reinitialisation
    reinit_board_poses: bool = True
    pnp_method: str = "iterative"  # 'ippe' = planar-safe (MOC fix)
    max_pnp_reprojection_px: float = 0.0  # 0 = no reprojection gate
    pnp_gate_radtan_only: bool = False    # the intrinsics are an untrusted seed for other models
    # Logging
    log_tag_coverage: bool = False


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class PerFrameStat:
    """Per-frame quality info for one camera."""
    frame_index: int
    tag_count: int
    obs_count: int
    kept: bool


@dataclass
class QualityGateResult:
    """Structured stats for logging and HTML report."""

    # Per-camera aggregate counts
    frames_before: dict[int, int] = field(default_factory=dict)
    frames_after: dict[int, int] = field(default_factory=dict)
    obs_before: dict[int, int] = field(default_factory=dict)
    obs_after: dict[int, int] = field(default_factory=dict)
    # Per-gate breakdown
    min_tag_dropped: dict[int, int] = field(default_factory=dict)
    pnp_rejected: dict[int, int] = field(default_factory=dict)
    # Sub-filter stats
    pose_filter_stats: PoseFilterStats | None = None
    corner_filter_stats: CornerFilterStats | None = None
    # Per-frame inlier counts (for HTML report plot)
    per_frame_stats: dict[int, list[PerFrameStat]] = field(default_factory=dict)

    def log_summary(self) -> None:
        """Log a unified summary of all quality gate results."""
        for cid in sorted(self.frames_before):
            fb = self.frames_before[cid]
            fa = self.frames_after.get(cid, 0)
            ob = self.obs_before.get(cid, 0)
            oa = self.obs_after.get(cid, 0)
            pct_f = 100 * fa / max(fb, 1)
            pct_o = 100 * oa / max(ob, 1)
            parts = [f"{fb}→{fa} frames ({pct_f:.1f}%)",
                     f"{ob}→{oa} obs ({pct_o:.1f}%)"]
            mt = self.min_tag_dropped.get(cid, 0)
            pr = self.pnp_rejected.get(cid, 0)
            if mt:
                parts.append(f"min-tag-dropped={mt}")
            if pr:
                parts.append(f"pnp-rejected={pr}")
            logger.info(f"  Quality gate cam {cid}: {', '.join(parts)}")


# ---------------------------------------------------------------------------
# Unified gate
# ---------------------------------------------------------------------------

def _board_corners_per_tag(board: Board) -> int:
    """Corners per tag: 4 for AprilGrid, 1 for chessboard."""
    n_tags = getattr(board, "_num_tags", 0)
    if n_tags and n_tags > 0:
        return 4
    return 1


def _num_expected_tags(board: Board) -> int:
    """Number of expected tags on the board."""
    n_tags = getattr(board, "_num_tags", 0)
    return n_tags if n_tags and n_tags > 0 else board.num_points()


def apply_quality_gates(
    all_frames: dict[int, list[FrameObservations]],
    board: Board,
    intrinsics: dict[int, np.ndarray],
    dist_coeffs: dict[int, np.ndarray],
    distortion_models: dict[int, str],
    config: QualityGateConfig,
    *,
    reference_camera: int = 0,
    pnp_frames: dict[int, list[FrameObservations]] | None = None,
) -> tuple[
    dict[int, list[FrameObservations]],
    dict[int, list],
    QualityGateResult,
]:
    """Apply all configured quality gates and return filtered frames + poses.

    Args:
        all_frames: cam_id → list of FrameObservations.
        board: Calibration board.
        intrinsics: cam_id → [fx, fy, cx, cy].
        dist_coeffs: cam_id → distortion coefficients.
        distortion_models: cam_id → "radtan" or "equidistant".
        config: Which gates to enable and their thresholds.
        reference_camera: Reference camera ID (for pooled-pose filter).
        pnp_frames: Optional full-corner frames for PnP initialization.
            When corner subsampling is used, ``all_frames`` contains
            subsampled observations (e.g. top_left only). PnP produces
            more accurate poses with all corners. If provided, PnP uses
            these frames (matched by timestamp to the filtered all_frames)
            instead of the subsampled ones.

    Returns:
        (filtered_frames, board_poses, result)
    """
    corners_per_tag = _board_corners_per_tag(board)
    num_expected = _num_expected_tags(board)
    min_pts = board.min_points_for_pose()

    result = QualityGateResult()

    # Record initial state
    for cid, frames in all_frames.items():
        result.frames_before[cid] = len(frames)
        result.obs_before[cid] = sum(len(f.observations) for f in frames)

    # Record per-frame stats BEFORE filtering (tag counts for plot)
    for cid, frames in all_frames.items():
        stats = []
        for fi, f in enumerate(frames):
            n_tags = len(_tag_ids(f.observations, corners_per_tag))
            stats.append(PerFrameStat(
                frame_index=fi, tag_count=n_tags,
                obs_count=len(f.observations), kept=True))
        result.per_frame_stats[cid] = stats

    # --- Gate 1: Per-frame dual filter (tag count + point count) ---
    _min_tc = config.min_tag_count or config.min_tags_per_frame
    _min_pc = config.min_point_count
    if _min_tc > 0:
        if _min_pc > 0:
            from mvis.detection.frame_quality import filter_frames_dual
            for cid in list(all_frames):
                before = len(all_frames[cid])
                all_frames[cid] = filter_frames_dual(
                    all_frames[cid],
                    min_tag_count=_min_tc,
                    min_point_count=_min_pc,
                    corners_per_tag=corners_per_tag)
                dropped = before - len(all_frames[cid])
                result.min_tag_dropped[cid] = dropped
        else:
            for cid in list(all_frames):
                before = len(all_frames[cid])
                all_frames[cid] = filter_frames_by_min_tags(
                    all_frames[cid], min_tags=_min_tc,
                    corners_per_tag=corners_per_tag)
                dropped = before - len(all_frames[cid])
                result.min_tag_dropped[cid] = dropped

    # --- Gate 2: Pooled-pose ref-frame filter ---
    if config.enable_pooled_pose_filter:
        all_frames, pose_stats = filter_ref_frames_by_pooled_tags(
            all_frames,
            reference_camera=reference_camera,
            num_expected_tags=num_expected,
            corners_per_tag=corners_per_tag,
            min_tag_floor=config.min_pose_tag_floor,
            min_pooled_tag_fraction=config.min_pose_tag_fraction,
            sync_tolerance_s=config.pose_sync_tolerance_s,
        )
        result.pose_filter_stats = pose_stats

    # --- Gate 3: Global corner observability filter ---
    if config.min_corner_views > 0:
        all_frames, corner_stats = filter_observations_by_min_views(
            all_frames, min_views=config.min_corner_views,
            min_points_for_pose=min_pts)
        result.corner_filter_stats = corner_stats

    # --- Gate 4: Per-camera corner observability filter ---
    if config.min_corner_views_per_camera > 0:
        all_frames = filter_observations_by_min_views_per_camera(
            all_frames, min_views=config.min_corner_views_per_camera,
            min_points_for_pose=min_pts)

    # --- Gate 5: PnP board pose reinitialisation + reprojection gate ---
    # If pnp_frames (full corners) are provided, use them for PnP to get
    # better initial poses. Match by timestamp to the filtered all_frames.
    board_poses: dict[int, list] = {}
    if config.reinit_board_poses:
        initializer = Initializer(board)
        for cid, frames in all_frames.items():
            if not frames or cid not in intrinsics:
                board_poses[cid] = []
                continue

            # Use full-corner frames for PnP if available
            frames_for_pnp = frames
            if pnp_frames and cid in pnp_frames:
                # Match full-corner frames to filtered frames by timestamp
                filtered_ts = {f.timestamp for f in frames}
                matched = [f for f in pnp_frames[cid]
                           if f.timestamp in filtered_ts]
                if len(matched) == len(frames):
                    frames_for_pnp = matched

            poses = initializer.init_board_poses(
                frames_for_pnp, intrinsics[cid], dist_coeffs[cid],
                distortion_models.get(cid, "radtan"),
                max_reprojection_px=(0.0 if config.pnp_gate_radtan_only
                                     and distortion_models.get(cid, "radtan") != "radtan"
                                     else config.max_pnp_reprojection_px),
                pnp_method=config.pnp_method)
            board_poses[cid] = poses
            # Count PnP rejections
            n_rejected = sum(1 for p in poses if p is None)
            n_min_pts_fail = sum(1 for f in frames
                                 if f.num_points < min_pts)
            # PnP rejected = None poses minus those that failed min_points
            result.pnp_rejected[cid] = max(0, n_rejected - n_min_pts_fail)
    else:
        for cid in all_frames:
            board_poses[cid] = []

    # Record final state
    for cid, frames in all_frames.items():
        result.frames_after[cid] = len(frames)
        result.obs_after[cid] = sum(len(f.observations) for f in frames)

    # Update per-frame kept/dropped status
    for cid, stats_list in result.per_frame_stats.items():
        kept_count = result.frames_after.get(cid, 0)
        # Mark frames beyond kept_count as dropped
        # (filters remove frames in order, so surviving frames are contiguous
        # only for min-tag; for corner filters, frames may be dropped anywhere)
        # Rebuild from the actual filtered frame set
        kept_timestamps = {f.timestamp for f in all_frames.get(cid, [])}
        for ps in stats_list:
            # Use original frame's timestamp to check if it survived
            orig_frames = []  # we don't have originals here
            # Approximate: if fewer frames survived, mark excess as dropped
            ps.kept = True  # default; we'll set properly below

    # Rebuild per-frame stats from final state with accurate kept flags
    # (the initial stats were recorded before filtering, now update)
    for cid in result.per_frame_stats:
        # After filtering, we can't map back by index perfectly.
        # Instead, rebuild from the filtered frames.
        final_stats = []
        filtered = all_frames.get(cid, [])
        for fi, f in enumerate(filtered):
            n_tags = len(_tag_ids(f.observations, corners_per_tag))
            pose_ok = (cid in board_poses and fi < len(board_poses[cid])
                       and board_poses[cid][fi] is not None)
            final_stats.append(PerFrameStat(
                frame_index=fi, tag_count=n_tags,
                obs_count=len(f.observations), kept=pose_ok))
        result.per_frame_stats[cid] = final_stats

    # --- Logging ---
    if config.log_tag_coverage:
        for cid, frames in all_frames.items():
            log_tag_coverage(frames, cid, num_expected, corners_per_tag)

    result.log_summary()
    return all_frames, board_poses, result
