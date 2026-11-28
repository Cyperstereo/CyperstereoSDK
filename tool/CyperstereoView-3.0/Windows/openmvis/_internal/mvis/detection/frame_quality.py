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
"""Detection-quality gating before observations enter the factor graph.

Two filters, applied in order between Stage 1 (init) and Stage 2 (per-camera
GTSAM optimization) of the calibration pipeline:

1. **Pooled per-pose tag-count filter** (``filter_ref_frames_by_pooled_tags``).

   Only ref-camera frames create a board-pose node ``B(i)`` that the
   trajectory hangs off. We require that *across all cameras combined*
   ≥ 20% of expected tags are observed near the ref timestamp:

       pooled_tags(t_i) = | union of tag-ids observed by any camera in
                           the sync window [t_i - tol, t_i + tol] |
       keep B(i)  iff  pooled_tags(t_i) >= max(min_tag_floor,
                                               ceil(0.20 × num_expected))

   This is more lenient than a per-camera threshold: a sparse ref-camera
   frame is fine if a synced non-ref camera saw plenty of tags. The
   intent is "is the *pose* well-constrained?" rather than "is each
   *frame* well-detected?".

   Non-ref camera frames are kept regardless (they only contribute
   observations against the surviving B(i)) — but they get the absolute
   floor filter applied via the per-frame ``min_points_for_pose``
   downstream.

2. **Per-corner observability filter** (``filter_observations_by_min_views``).
   Drops observations of any board point that has fewer than ``min_views``
   total observations across *all* (camera, frame) pairs. View counts are
   pooled across cameras; a corner observed once each by 5 different
   cameras counts as 5 views.

   Reasoning: a corner observed only 1-2 times provides no redundant
   constraint against detection noise. Removing it costs nothing (the
   board geometry is fixed; the corner's 3D position is known) and
   reduces the chance that a single mis-detection biases intrinsics.
"""

from __future__ import annotations

import bisect
import logging
import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from mvis.core.types import FrameObservations, Observation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pooled-per-pose ref-frame filter
# ---------------------------------------------------------------------------

@dataclass
class PoseFilterStats:
    """Result of the pooled per-pose ref-frame filter."""

    reference_camera: int
    threshold_tags: int
    n_ref_in: int
    n_ref_kept: int
    n_ref_dropped: int
    sync_tolerance_s: float
    dropped_pooled_counts: Counter = field(default_factory=Counter)


def _tag_ids(obs: list[Observation], corners_per_tag: int) -> set[int]:
    """Distinct tag ids in an observation list."""
    if corners_per_tag <= 1:
        return {o.point_id for o in obs}
    return {o.point_id // corners_per_tag for o in obs}


def filter_ref_frames_by_pooled_tags(
    frames_by_cam: dict[int, list[FrameObservations]],
    *,
    reference_camera: int,
    num_expected_tags: int,
    corners_per_tag: int = 4,
    min_tag_floor: int = 4,
    min_pooled_tag_fraction: float = 0.20,
    sync_tolerance_s: float | None = None,
) -> tuple[dict[int, list[FrameObservations]], PoseFilterStats]:
    """Drop ref-camera frames whose pose would be poorly constrained.

    For each ref-camera frame at time ``t_i``, pool the unique tag-ids
    observed by *any* camera within the sync window
    ``[t_i - tol, t_i + tol]``. If the union has fewer than
    ``max(min_tag_floor, ceil(min_pooled_tag_fraction × num_expected_tags))``
    tags, drop the ref frame. The pose ``B(i)`` will not be created.

    Non-ref camera frames are passed through unchanged — they only
    contribute observations against ref poses that *do* survive.

    Args:
        sync_tolerance_s: half-window of "synced" non-ref frames around
            each ref timestamp. ``None`` = auto-pick as half the median
            inter-frame interval of the ref camera.
    """
    if reference_camera not in frames_by_cam:
        # No ref camera in input — pass through.
        stats = PoseFilterStats(
            reference_camera=reference_camera,
            threshold_tags=0, n_ref_in=0, n_ref_kept=0, n_ref_dropped=0,
            sync_tolerance_s=0.0,
        )
        return frames_by_cam, stats

    threshold = max(min_tag_floor,
                    math.ceil(min_pooled_tag_fraction * num_expected_tags))
    ref_frames = frames_by_cam[reference_camera]

    # Auto sync tolerance from ref-camera median frame interval.
    if sync_tolerance_s is None:
        ts = sorted(f.timestamp for f in ref_frames)
        if len(ts) >= 2:
            diffs = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
            diffs.sort()
            median_dt = diffs[len(diffs) // 2]
            sync_tolerance_s = max(median_dt / 2.0, 1e-3)
        else:
            sync_tolerance_s = 0.05  # 50 ms fallback

    # Index non-ref frames by timestamp for binary search.
    non_ref_index: dict[int, list[tuple[float, FrameObservations]]] = {}
    for cid, frames in frames_by_cam.items():
        if cid == reference_camera:
            continue
        non_ref_index[cid] = sorted(
            ((f.timestamp, f) for f in frames), key=lambda x: x[0])

    kept_ref: list[FrameObservations] = []
    dropped_pool_hist: Counter = Counter()

    for ref_f in ref_frames:
        t = ref_f.timestamp
        pooled = _tag_ids(ref_f.observations, corners_per_tag)
        for cid, sorted_frames in non_ref_index.items():
            ts_list = [t for t, _ in sorted_frames]
            lo = bisect.bisect_left(ts_list, t - sync_tolerance_s)
            hi = bisect.bisect_right(ts_list, t + sync_tolerance_s)
            for _, f in sorted_frames[lo:hi]:
                pooled |= _tag_ids(f.observations, corners_per_tag)
        if len(pooled) >= threshold:
            kept_ref.append(ref_f)
        else:
            dropped_pool_hist[len(pooled)] += 1

    out = dict(frames_by_cam)
    out[reference_camera] = kept_ref

    stats = PoseFilterStats(
        reference_camera=reference_camera,
        threshold_tags=threshold,
        n_ref_in=len(ref_frames),
        n_ref_kept=len(kept_ref),
        n_ref_dropped=len(ref_frames) - len(kept_ref),
        sync_tolerance_s=sync_tolerance_s,
        dropped_pooled_counts=dropped_pool_hist,
    )
    if stats.n_ref_dropped > 0:
        example = ", ".join(f"{n}×{cnt}"
                              for cnt, n in sorted(dropped_pool_hist.items()))
        logger.info(
            f"Pooled-pose filter: ref cam {reference_camera} "
            f"kept {stats.n_ref_kept}/{stats.n_ref_in} "
            f"(dropped {stats.n_ref_dropped} with pooled tag counts: {example}); "
            f"threshold ≥{threshold} tags, sync window ±{sync_tolerance_s*1000:.0f}ms"
        )
    else:
        logger.info(
            f"Pooled-pose filter: ref cam {reference_camera} "
            f"kept all {stats.n_ref_kept} ref frames "
            f"(threshold ≥{threshold} tags pooled across cameras, "
            f"sync window ±{sync_tolerance_s*1000:.0f}ms)"
        )
    return out, stats


# ---------------------------------------------------------------------------
# Per-corner observability filter
# ---------------------------------------------------------------------------

@dataclass
class CornerFilterStats:
    """Result of the per-corner observability filter."""

    n_unique_corners: int
    n_corners_dropped: int
    min_views: int
    obs_in: int
    obs_out: int
    drop_view_counts: Counter = field(default_factory=Counter)
    n_frames_in: int = 0
    n_frames_out: int = 0


def filter_observations_by_min_views(
    frames_by_cam: dict[int, list[FrameObservations]],
    *,
    min_views: int = 5,
    min_points_for_pose: int = 4,
) -> tuple[dict[int, list[FrameObservations]], CornerFilterStats]:
    """Drop observations of corners with fewer than ``min_views`` total views.

    Views are pooled across all (camera, frame) pairs — a corner observed
    once from each of 5 cameras counts as 5 views.

    Frames that fall below ``min_points_for_pose`` after the drop are
    removed entirely.
    """
    counter: Counter[int] = Counter()
    obs_in = 0
    n_frames_in = 0
    for frames in frames_by_cam.values():
        n_frames_in += len(frames)
        for f in frames:
            obs_in += len(f.observations)
            for o in f.observations:
                counter[o.point_id] += 1

    drop_set: set[int] = {pid for pid, n in counter.items() if n < min_views}
    drop_view_counts: Counter = Counter()
    for pid in drop_set:
        drop_view_counts[counter[pid]] += 1

    out: dict[int, list[FrameObservations]] = {}
    obs_out = 0
    n_frames_out = 0
    for cam_id, frames in frames_by_cam.items():
        new_frames: list[FrameObservations] = []
        for f in frames:
            kept = [o for o in f.observations if o.point_id not in drop_set]
            if len(kept) >= min_points_for_pose:
                new_frames.append(FrameObservations(
                    camera_id=f.camera_id, timestamp=f.timestamp,
                    observations=kept, image_size=f.image_size,
                ))
                obs_out += len(kept)
        out[cam_id] = new_frames
        n_frames_out += len(new_frames)

    stats = CornerFilterStats(
        n_unique_corners=len(counter),
        n_corners_dropped=len(drop_set),
        min_views=min_views,
        obs_in=obs_in, obs_out=obs_out,
        drop_view_counts=drop_view_counts,
        n_frames_in=n_frames_in, n_frames_out=n_frames_out,
    )
    if drop_set:
        logger.info(
            f"Corner observability filter: {len(counter)} unique corners, "
            f"dropped {len(drop_set)} with <{min_views} views "
            f"(drop view-count histogram: "
            f"{dict(sorted(drop_view_counts.items()))}); "
            f"obs {obs_in} → {obs_out}, frames {n_frames_in} → {n_frames_out}"
        )
    else:
        logger.info(
            f"Corner observability filter: {len(counter)} unique corners, "
            f"all have ≥{min_views} views — no drops"
        )
    return out, stats


# ---------------------------------------------------------------------------
# Per-camera corner observability filter
# ---------------------------------------------------------------------------

def filter_observations_by_min_views_per_camera(
    frames_by_cam: dict[int, list[FrameObservations]],
    *,
    min_views: int = 3,
    min_points_for_pose: int = 4,
) -> dict[int, list[FrameObservations]]:
    """Drop corners seen fewer than ``min_views`` times *within each camera*.

    Unlike ``filter_observations_by_min_views`` (which pools across all
    cameras), this checks each camera independently.  A corner observed
    5× by cam0 but 1× by cam1 will be kept for cam0 and dropped for cam1.

    Critical for Stage 2 per-camera optimization where each camera's
    factor graph is independent — a corner observed only 1–2 times in a
    single camera's graph provides almost no constraint on intrinsics.
    """
    out: dict[int, list[FrameObservations]] = {}
    for cam_id, frames in frames_by_cam.items():
        # Count per-corner views for THIS camera only
        counter: Counter[int] = Counter()
        for f in frames:
            for o in f.observations:
                counter[o.point_id] += 1

        drop_set = {pid for pid, n in counter.items() if n < min_views}
        if not drop_set:
            out[cam_id] = list(frames)
            continue

        new_frames: list[FrameObservations] = []
        for f in frames:
            kept = [o for o in f.observations if o.point_id not in drop_set]
            if len(kept) >= min_points_for_pose:
                new_frames.append(FrameObservations(
                    camera_id=f.camera_id, timestamp=f.timestamp,
                    observations=kept, image_size=f.image_size,
                ))
        out[cam_id] = new_frames
        n_dropped = len(drop_set)
        n_obs_before = sum(len(f.observations) for f in frames)
        n_obs_after = sum(len(f.observations) for f in new_frames)
        logger.info(
            f"  Camera {cam_id}: per-camera corner filter dropped "
            f"{n_dropped} corners with <{min_views} views "
            f"(obs {n_obs_before} → {n_obs_after}, "
            f"frames {len(frames)} → {len(new_frames)})"
        )
    return out


# ---------------------------------------------------------------------------
# Per-frame minimum tag count filter
# ---------------------------------------------------------------------------

def filter_frames_by_min_tags(
    frames: list[FrameObservations],
    *,
    min_tags: int = 4,
    corners_per_tag: int = 4,
) -> list[FrameObservations]:
    """Keep only frames with at least ``min_tags`` unique tags.

    For AprilGrid with top_left subsampling (corners_per_tag=1),
    each observation is a unique tag so this equals min observation count.
    """
    kept = []
    for f in frames:
        n_tags = len(_tag_ids(f.observations, corners_per_tag))
        if n_tags >= min_tags:
            kept.append(f)
    n_dropped = len(frames) - len(kept)
    if n_dropped > 0:
        logger.info(
            f"  Min-tag filter: {len(frames)} → {len(kept)} frames "
            f"(dropped {n_dropped} with <{min_tags} tags)"
        )
    return kept


def filter_frames_dual(
    frames: list[FrameObservations],
    *,
    min_tag_count: int = 4,
    min_point_count: int = 6,
    corners_per_tag: int = 4,
) -> list[FrameObservations]:
    """Keep frames with sufficient tags AND observation points.

    A frame is kept only if BOTH:
      - tag_count >= min_tag_count (spatial spread)
      - len(observations) >= min_point_count (data quantity)

    This dual check ensures:
      - Enough tags for spatial diversity (prevents degenerate PnP from
        clustered corners of 1-2 tags)
      - Enough observation points for robust optimization (adaptive
        subsampling gives sparse frames more corners per tag)
    """
    kept = []
    for f in frames:
        n_tags = len(_tag_ids(f.observations, corners_per_tag))
        n_pts = len(f.observations)
        if n_tags >= min_tag_count and n_pts >= min_point_count:
            kept.append(f)
    n_dropped = len(frames) - len(kept)
    if n_dropped > 0:
        logger.info(
            f"  Min-tag/point filter: {len(frames)} → {len(kept)} frames "
            f"(dropped {n_dropped} with <{min_tag_count} tags or "
            f"<{min_point_count} pts)"
        )
    return kept


# ---------------------------------------------------------------------------
# Tag coverage statistics
# ---------------------------------------------------------------------------

def filter_frames_by_tag_spread(
    frames: list[FrameObservations],
    *,
    min_hull_fraction: float = 0.01,
    corners_per_tag: int = 4,
) -> list[FrameObservations]:
    """Keep only frames where detected tags cover enough of the image.

    Computes the convex hull of tag centers as a fraction of image area.
    Frames where all tags are clustered in a small region (e.g., board
    at FOV edge) have degenerate PnP geometry — especially for fisheye.

    Args:
        frames: Frame observations for one camera.
        min_hull_fraction: Minimum convex hull area / image area. Default
            0.01 (1%) — extremely clustered detections.
        corners_per_tag: Corners per tag (4 for AprilGrid, 1 for top_left).

    Returns:
        Filtered frame list.
    """
    from scipy.spatial import ConvexHull

    kept = []
    n_dropped = 0
    for f in frames:
        if not f.observations or f.image_size is None:
            kept.append(f)
            continue

        # Compute tag centers
        tag_pixels: dict[int, list] = {}
        for o in f.observations:
            tid = o.point_id // corners_per_tag if corners_per_tag > 1 else o.point_id
            tag_pixels.setdefault(tid, []).append(o.pixel)
        centers = [np.mean(pts, axis=0) for pts in tag_pixels.values()]

        if len(centers) < 3:
            kept.append(f)  # can't compute hull, let min_tags handle it
            continue

        w, h = f.image_size
        img_area = w * h
        if img_area <= 0:
            kept.append(f)
            continue

        try:
            hull = ConvexHull(np.array(centers))
            hull_fraction = hull.volume / img_area  # 2D: volume = area
        except Exception:
            hull_fraction = 0.0

        if hull_fraction >= min_hull_fraction:
            kept.append(f)
        else:
            n_dropped += 1

    if n_dropped > 0:
        logger.info(
            f"  Tag-spread filter: {len(frames)} → {len(kept)} frames "
            f"(dropped {n_dropped} with hull < {min_hull_fraction*100:.1f}% of image)")
    return kept


def log_tag_coverage(
    frames: list[FrameObservations],
    camera_id: int,
    num_expected_tags: int,
    corners_per_tag: int = 4,
) -> None:
    """Log per-tag observation coverage for diagnostics."""
    counter: Counter[int] = Counter()
    for f in frames:
        for tid in _tag_ids(f.observations, corners_per_tag):
            counter[tid] += 1

    if not counter:
        logger.info(f"  Camera {camera_id}: no tags observed")
        return

    n_unique = len(counter)
    counts = sorted(counter.values())
    min_v, med_v, max_v = counts[0], counts[len(counts) // 2], counts[-1]
    sparse = [tid for tid, n in counter.items() if n < 3]
    msg = (f"  Camera {camera_id}: {n_unique}/{num_expected_tags} tags, "
           f"views/tag: min={min_v} med={med_v} max={max_v}")
    if sparse:
        msg += f", {len(sparse)} tags with <3 views"
    logger.info(msg)
