# OpenMVIS: A Multi-Visual-Inertial Sensor Calibration Benchmark
# Copyright (C) 2026 Yulin Yang
# Copyright (C) 2026 Yuantai-Z
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
# SPDX-FileCopyrightText: 2026 Yuantai-Z
# SPDX-License-Identifier: GPL-3.0-only
"""Frame subsampling for calibration.

Selects a subset of frames that provide diverse viewpoints while avoiding
redundant near-identical frames:

  * `subsample_frames_by_hz` / `select_frame_indices_by_hz`: thin each camera to a
    target rate by nearest-timestamp matching (the pipelines' default).
  * `subsample_by_detection_diversity`: keep frames whose detected corners moved,
    from the 2-D detections alone (no PnP).
  * `co_observed_selection` / `nearest_ref`: pair non-reference cameras with the
    reference camera's frames.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def subsample_by_detection_diversity(
    frames: list[tuple],
    observations_per_frame: list[list],
    min_centroid_shift_px: float = 15.0,
    min_rotation_proxy_px: float = 10.0,
    min_time_gap: float = 0.1,
    image_size: tuple[int, int] | None = None,
    min_centroid_shift_frac: float = 0.02,
    min_rotation_proxy_frac: float = 0.012,
) -> tuple[list[int], list[int]]:
    """Select frames where the board moved significantly based on detected corners.

    Uses centroid displacement and corner spread change as proxy for board motion.
    No PnP needed — works directly on 2D detections.

    Thresholds are resolution-adaptive: when ``image_size`` is provided,
    ``min_centroid_shift_frac`` and ``min_rotation_proxy_frac`` (fractions of
    the image diagonal) override the absolute pixel thresholds.  This ensures
    consistent filtering across different camera resolutions (e.g. 640×480
    vs 3840×2880).

    Args:
        frames: List of (image_or_path, timestamp) tuples.
        observations_per_frame: List of observation lists (one per frame).
        min_centroid_shift_px: Minimum centroid displacement in pixels
            (used when image_size is not provided).
        min_rotation_proxy_px: Minimum spread change in pixels
            (used when image_size is not provided).
        min_time_gap: Minimum time between selected frames (seconds).
        image_size: (width, height) of the image.  When provided,
            fractional thresholds are used instead of absolute pixels.
        min_centroid_shift_frac: Minimum centroid shift as fraction of
            image diagonal (default 0.02 = 2%).
        min_rotation_proxy_frac: Minimum spread change as fraction of
            image diagonal (default 0.012 = 1.2%).

    Returns:
        (selected_frame_indices, rejected_indices).
    """
    if len(frames) <= 3:
        return list(range(len(frames))), []

    # Compute resolution-adaptive thresholds
    if image_size is not None:
        diag = np.hypot(float(image_size[0]), float(image_size[1]))
        shift_thresh = diag * min_centroid_shift_frac
        spread_thresh = diag * min_rotation_proxy_frac
    else:
        shift_thresh = min_centroid_shift_px
        spread_thresh = min_rotation_proxy_px

    def _frame_signature(observations):
        """Compute centroid + spread of detected corners."""
        if not observations:
            return None
        pixels = np.array([obs.pixel for obs in observations])
        centroid = pixels.mean(axis=0)
        spread = pixels.std(axis=0).mean()  # proxy for board size/orientation
        return centroid, spread

    selected = [0]  # always keep first frame with detections
    # Find first frame with detections
    for i in range(len(frames)):
        if observations_per_frame[i]:
            selected = [i]
            break

    last_sig = _frame_signature(observations_per_frame[selected[0]])
    last_time = frames[selected[0]][1]

    for i in range(selected[0] + 1, len(frames)):
        obs = observations_per_frame[i]
        if not obs:
            continue

        t = frames[i][1]
        if t - last_time < min_time_gap:
            continue

        sig = _frame_signature(obs)
        if sig is None or last_sig is None:
            selected.append(i)
            last_sig = sig
            last_time = t
            continue

        centroid_shift = np.linalg.norm(sig[0] - last_sig[0])
        spread_change = abs(sig[1] - last_sig[1])

        if centroid_shift > shift_thresh or spread_change > spread_thresh:
            selected.append(i)
            last_sig = sig
            last_time = t

    # Always include the last frame with detections — a NO-OP when the main
    # loop already selected it. The old `and i not in selected` walked PAST
    # an already-selected last frame and appended the next-lower detected
    # index, making `selected` non-monotonic (frames out of timestamp order)
    # on recordings where the final frame passed the diversity test — caught
    # by the 2c-8 selection shadow (regression-tiers vetting, 2026-09-09).
    for i in range(len(frames) - 1, -1, -1):
        if observations_per_frame[i]:
            if i not in selected:
                selected.append(i)
            break

    rejected = [i for i in range(len(frames)) if i not in set(selected)]
    return selected, rejected


def select_frame_indices_by_hz(
    all_frames: dict,
    target_hz: float | dict = 5.0,
) -> dict:
    """Select source-frame indices for a per-camera target rate.

    Cameras whose effective rate is already at or below ``target_hz * 1.2``
    are passed through unchanged.  Uses nearest-timestamp matching for
    accurate subsampling even when source_hz / target_hz is not an integer.

    Args:
        all_frames: ``{cam_id: [FrameObservations, ...]}``.
        target_hz: Target rate — float for uniform, or
            ``{cam_id: hz}`` for per-camera rates.

    Returns:
        ``{cam_id: [source_frame_index, ...]}``.  Every index is relative to
        the corresponding input list in ``all_frames``.
    """
    def _resolve(hz, cid):
        return hz.get(cid, 5.0) if isinstance(hz, dict) else float(hz)

    # One grid origin for ALL cameras: the earliest timestamp of any camera.
    # A per-camera origin (each camera's own ts[0]) shifts the grids by a
    # whole frame whenever one stream starts a frame earlier than the other,
    # so synchronized cameras select DIFFERENT instants and every non-ref
    # frame is then interpolated between ref poses 1/target_hz apart (T265
    # stereo bags 2/3/5/10/12/14: cam3 P95 10-16 px instead of ~2 px).
    # Single-camera calls are unchanged (origin == that camera's ts[0]).
    starts = [f[0].timestamp for f in all_frames.values() if len(f) > 0]
    t_origin = min(starts) if starts else 0.0

    selected_by_camera = {}
    for cid, frames in all_frames.items():
        cam_hz = _resolve(target_hz, cid)
        if len(frames) < 2:
            selected_by_camera[cid] = list(range(len(frames)))
            continue
        ts = [f.timestamp for f in frames]
        duration = ts[-1] - ts[0]
        current_hz = (len(ts) - 1) / duration if duration > 0 else 0
        if current_hz > cam_hz * 1.2:
            ts_np = np.array(ts)
            ideal_times = np.arange(t_origin, ts[-1], 1.0 / cam_hz)
            # Grid points more than half a frame before this camera's first
            # frame have no counterpart here; without this they would all
            # collapse onto frame 0 and add an unpaired frame.
            half_frame = 0.5 * float(np.median(np.diff(ts_np)))
            ideal_times = ideal_times[ideal_times >= ts_np[0] - half_frame]
            indices = sorted(
                {int(np.argmin(np.abs(ts_np - t))) for t in ideal_times})
            selected_by_camera[cid] = indices
        else:
            selected_by_camera[cid] = list(range(len(frames)))
    return selected_by_camera


def subsample_frames_by_hz(
    all_frames: dict,
    target_hz: float | dict = 5.0,
) -> dict:
    """Subsample per-camera FrameObservations to a target rate.

    This compatibility API keeps returning frame objects.  Call
    :func:`select_frame_indices_by_hz` when a parallel data track (for example
    board poses) must be selected with the exact same source indices.
    """
    selected_by_camera = select_frame_indices_by_hz(
        all_frames, target_hz=target_hz)
    subsampled = {}
    for cid, frames in all_frames.items():
        indices = selected_by_camera[cid]
        subsampled[cid] = [frames[i] for i in indices]
        if len(indices) < len(frames) and len(frames) >= 2:
            ts = [f.timestamp for f in frames]
            duration = ts[-1] - ts[0]
            current_hz = (len(ts) - 1) / duration if duration > 0 else 0
            cam_hz = (target_hz.get(cid, 5.0)
                      if isinstance(target_hz, dict) else float(target_hz))
            logger.info(
                f"  cam{cid}: {len(frames)} -> {len(subsampled[cid])} "
                f"({current_hz:.1f} -> {cam_hz:.0f}Hz)")
    return subsampled


def nearest_ref(ts, ref_ts) -> tuple[np.ndarray, np.ndarray]:
    """Index of the nearest entry of the sorted, non-empty `ref_ts` for each time in `ts`,
    and its distance |ts - ref_ts[k]|. Of two equally near neighbours the lower index wins."""
    ts = np.asarray(ts)
    j = np.clip(np.searchsorted(ref_ts, ts), 1, max(len(ref_ts) - 1, 1))
    lo = np.maximum(j - 1, 0)
    hi = np.minimum(j, len(ref_ts) - 1)
    d_lo = np.abs(ts - ref_ts[lo])
    d_hi = np.abs(ts - ref_ts[hi])
    take_hi = d_hi < d_lo
    return np.where(take_hi, hi, lo), np.where(take_hi, d_hi, d_lo)


def _near_any(ts: np.ndarray, ref_ts: np.ndarray, tol: float) -> np.ndarray:
    """Boolean mask: ts[i] has some ref_ts within ``tol`` (ref_ts sorted)."""
    if len(ref_ts) == 0 or len(ts) == 0:
        return np.zeros(len(ts), dtype=bool)
    return nearest_ref(ts, ref_ts)[1] <= tol


def co_observed_selection(
    frames: dict,
    independent_idx: dict,
    ref_cam: int,
    frame_period: dict,
    min_keep_frac: float = 0.5,
) -> dict:
    """Joint diversity selection for synchronized cameras (no time offset).

    The joint solve has pose nodes on the reference camera only; a non-ref
    frame with no ref frame at the same instant is interpolated between ref
    poses that may be >= 400 ms apart. So, starting from each camera's own
    diversity picks (``independent_idx``):

    * the selected INSTANTS are the union of every camera's picks;
    * the ref camera keeps its frame at each instant (when it has one);
    * a non-ref camera keeps its frame at an instant only if the ref kept a
      frame within half a source frame period of it.

    A camera that would keep fewer than ``min_keep_frac`` of its own picks
    (e.g. unsynchronized streams, little co-visibility) falls back to its
    independent selection, so a rig is never starved.

    Args:
        frames: ``{cam_id: [FrameObservations]}`` sorted by timestamp (the
            rate-filtered candidates).
        independent_idx: ``{cam_id: [index into frames[cam_id]]}``.
        ref_cam: reference camera id.
        frame_period: ``{cam_id: source frame period (s)}``.

    Returns:
        ``{cam_id: sorted [index into frames[cam_id]]}``.
    """
    if ref_cam not in frames or len(frames) < 2:
        return {c: sorted(v) for c, v in independent_idx.items()}
    ts = {c: np.array([f.timestamp for f in fr]) for c, fr in frames.items()}
    instants = np.unique(np.concatenate(
        [ts[c][np.asarray(ix, dtype=int)] for c, ix in independent_idx.items()
         if len(ix)] or [np.array([])]))
    half = {c: 0.5 * frame_period.get(c, 0.0) for c in frames}

    ref_keep = np.flatnonzero(_near_any(ts[ref_cam], instants, half[ref_cam]))
    ref_kept_ts = ts[ref_cam][ref_keep]
    out = {ref_cam: ref_keep.tolist()}
    for c in frames:
        if c == ref_cam:
            continue
        tol = max(half[c], half[ref_cam])
        mask = (_near_any(ts[c], instants, half[c])
                & _near_any(ts[c], ref_kept_ts, tol))
        keep = np.flatnonzero(mask).tolist()
        n_ind = len(independent_idx.get(c, []))
        if len(keep) < min_keep_frac * n_ind:
            logger.warning(
                f"  cam{c}: only {len(keep)}/{n_ind} diversity frames are "
                f"co-observed with ref cam{ref_cam} — keeping its "
                f"independent selection")
            keep = sorted(independent_idx.get(c, []))
        out[c] = keep
    return out
