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
"""Frame selection strategies for calibration initialization.

Initialization quality scales with frame *diversity* (image position,
apparent size, orientation), not just frame count. Picking 50 well-spread
frames typically beats 500 redundant ones.

Two strategies are exposed:

- ``uniform_in_time``: equally-spaced indices over the time-ordered
  sequence. This matches what ``pipeline.py`` already does — no surprise
  in behavior when this strategy is selected.

- ``diverse``: bin frames into a 3-D grid over (image position,
  apparent size, orientation) and sample uniformly across non-empty
  bins. Better init focal-length conditioning, especially for fisheye
  and wide-angle lenses.

The selectors operate on ``FrameObservations`` and return a list of
the same type. They never construct new frames, only pick a subset.
"""

from __future__ import annotations

import logging

import numpy as np

from mvis.core.types import FrameObservations

logger = logging.getLogger("mvis.calibration.init_frame_selector")


def select_uniform_in_time(
    frames: list[FrameObservations],
    n_keep: int,
) -> list[FrameObservations]:
    """Return ``n_keep`` frames spaced uniformly across the time-ordered list."""
    if n_keep <= 0 or len(frames) <= n_keep:
        return list(frames)
    indices = np.linspace(0, len(frames) - 1, n_keep, dtype=int)
    return [frames[i] for i in indices]


def _frame_metrics(
    frame: FrameObservations,
    image_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Compute (cx_norm, cy_norm, size_norm, orientation) for one frame.

    Coordinates are normalized to [0, 1] over image dimensions; orientation
    is the angle (radians) of the first principal axis of the detected
    point cloud, wrapped to [0, pi).
    """
    pts = frame.pixels  # Nx2
    w, h = image_size
    if w <= 0 or h <= 0 or pts.shape[0] < 2:
        return 0.5, 0.5, 0.0, 0.0

    centroid = pts.mean(axis=0)
    cx_norm = float(np.clip(centroid[0] / w, 0.0, 1.0))
    cy_norm = float(np.clip(centroid[1] / h, 0.0, 1.0))

    bbox = pts.max(axis=0) - pts.min(axis=0)
    diag = float(np.linalg.norm(bbox))
    img_diag = float(np.hypot(w, h))
    size_norm = float(np.clip(diag / img_diag, 0.0, 1.0)) if img_diag > 0 else 0.0

    centered = pts - centroid
    cov = centered.T @ centered
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal = eigvecs[:, -1]
        angle = float(np.arctan2(principal[1], principal[0]))
    except np.linalg.LinAlgError:
        angle = 0.0
    if angle < 0:
        angle += np.pi
    if angle >= np.pi:
        angle -= np.pi

    return cx_norm, cy_norm, size_norm, angle


def select_diverse(
    frames: list[FrameObservations],
    n_keep: int,
    image_size: tuple[int, int],
    n_pos_bins: int = 4,
    n_size_bins: int = 4,
    n_orient_bins: int = 4,
) -> list[FrameObservations]:
    """Pick ``n_keep`` frames maximizing diversity in position/size/orientation.

    Algorithm:
        1. Compute (centroid_x, centroid_y, size, orientation) per frame.
        2. Quantize each into bins:
             position    -> n_pos_bins x n_pos_bins grid
             size        -> n_size_bins bins
             orientation -> n_orient_bins bins (over [0, pi))
        3. Round-robin across non-empty bins until ``n_keep`` frames are picked.

    Falls back to ``select_uniform_in_time`` if ``n_keep >= len(frames)`` or
    no valid metrics could be computed.
    """
    if n_keep <= 0 or len(frames) <= n_keep:
        return list(frames)

    metrics = [_frame_metrics(f, image_size) for f in frames]
    cx = np.array([m[0] for m in metrics])
    cy = np.array([m[1] for m in metrics])
    sz = np.array([m[2] for m in metrics])
    th = np.array([m[3] for m in metrics])

    cx_b = np.clip((cx * n_pos_bins).astype(int), 0, n_pos_bins - 1)
    cy_b = np.clip((cy * n_pos_bins).astype(int), 0, n_pos_bins - 1)
    sz_b = np.clip((sz * n_size_bins).astype(int), 0, n_size_bins - 1)
    th_b = np.clip((th / np.pi * n_orient_bins).astype(int), 0, n_orient_bins - 1)

    bin_to_indices: dict[tuple[int, int, int, int], list[int]] = {}
    for i, key in enumerate(zip(cx_b, cy_b, sz_b, th_b)):
        bin_to_indices.setdefault(tuple(int(x) for x in key), []).append(i)

    if not bin_to_indices:
        return select_uniform_in_time(frames, n_keep)

    bin_keys = sorted(bin_to_indices.keys())
    rng = np.random.default_rng(seed=0)
    for k in bin_keys:
        rng.shuffle(bin_to_indices[k])

    picked: list[int] = []
    cursors = {k: 0 for k in bin_keys}
    while len(picked) < n_keep:
        progressed = False
        for k in bin_keys:
            if cursors[k] < len(bin_to_indices[k]):
                picked.append(bin_to_indices[k][cursors[k]])
                cursors[k] += 1
                progressed = True
                if len(picked) >= n_keep:
                    break
        if not progressed:
            break

    picked.sort()
    logger.info(
        "diverse selector: %d frames -> %d picks across %d non-empty bins "
        "(grid %dx%dx%dx%d)",
        len(frames), len(picked), len(bin_keys),
        n_pos_bins, n_pos_bins, n_size_bins, n_orient_bins,
    )
    return [frames[i] for i in picked]


def select_init_frames(
    frames: list[FrameObservations],
    n_keep: int,
    image_size: tuple[int, int],
    strategy: str = "uniform",
) -> list[FrameObservations]:
    """Top-level entry point. ``strategy`` is "uniform" or "diverse"."""
    if strategy == "diverse":
        return select_diverse(frames, n_keep, image_size)
    if strategy != "uniform":
        logger.warning(
            "Unknown init_frame_selection=%r, falling back to uniform", strategy)
    return select_uniform_in_time(frames, n_keep)
