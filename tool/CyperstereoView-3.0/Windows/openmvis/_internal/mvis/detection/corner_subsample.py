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
"""Per-tag corner subsampling for AprilGrid observations.

Rationale
---------
An AprilGrid detection emits 4 corners per tag. Feeding all 4 into the
factor graph is accurate but slow; for regression and speed experiments
we sometimes want to keep only a subset (e.g. 1 corner per tag for a
~4x reduction in factor count).

The filter operates on already-detected ``Observation`` lists: the
detector still runs on the full image (it needs all 4 corners to decode
the tag), and the 3D board geometry is unchanged — only the measurement
set passed downstream is thinned.

Corner index convention (follows aprilgrid / ``AprilTagBoard``)::

    point_id = tag_idx * 4 + corner_idx

    corner_idx 0 = bottom-left   ("bl")
    corner_idx 1 = bottom-right  ("br")
    corner_idx 2 = top-right     ("tr")
    corner_idx 3 = top-left      ("tl", "top_left")

The labelling is *tag-intrinsic* — it follows the encoded bit pattern
on the tag face, so "top_left" means the same physical corner
regardless of how the board is rotated in the image.

Modes
-----
    "all"       — keep every observation (default; no-op).
    "top_left"  — keep only corner_idx == 3 (alias: "tl").
    "bl" / "br" / "tr" / "tl" — single named corner.
    "two_diag"  — keep corners 0 and 2 (bottom-left + top-right).
    "two_anti"  — keep corners 1 and 3 (bottom-right + top-left).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from mvis.core.types import Observation

_SINGLE_CORNER = {
    "bl": 0,
    "br": 1,
    "tr": 2,
    "tl": 3,
    "top_left": 3,
    "bottom_left": 0,
    "bottom_right": 1,
    "top_right": 2,
}

_MULTI_CORNER = {
    "two_diag": (0, 2),    # bl + tr
    "two_anti": (1, 3),    # br + tl
}


def corner_mask(ids, mode: str) -> np.ndarray:
    """Per-corner keep mask on AprilGrid point ids (tag * 4 + corner) for `mode`.

    The one dispatch behind `filter_observations`, the cam-IMU solve and the 3-D
    board plot. Raises ValueError on an unknown mode.
    """
    if mode == "all":
        return np.ones(len(ids), bool)
    corner = np.asarray(ids, int) % 4
    if mode in _SINGLE_CORNER:
        return corner == _SINGLE_CORNER[mode]
    if mode in _MULTI_CORNER:
        return np.isin(corner, _MULTI_CORNER[mode])
    raise ValueError(
        f"Unknown corner_subsample mode: {mode!r}. "
        f"Valid: 'all', 'top_left'/'tl', 'bl', 'br', 'tr', 'two_diag', 'two_anti'."
    )


def corners_per_tag(mode: str) -> float:
    """Corners kept per AprilGrid tag under `mode` (4, 2 or 1; 1 for an unknown mode),
    the subsample factor of the timing model."""
    if mode == "all":
        return 4.0
    if mode in _MULTI_CORNER:
        return 2.0
    return 1.0


def filter_observations(obs: Iterable[Observation], mode: str) -> list[Observation]:
    """Return a subset of AprilGrid ``Observation``s per the given mode.

    For non-AprilGrid boards (chessboards, etc.) this is a no-op when
    ``mode == "all"``; other modes will still filter by ``point_id % 4``
    which is not meaningful for those boards, so don't pass
    ``corner_subsample != "all"`` to a non-AprilGrid calibration.
    """
    obs_list = list(obs)
    if mode == "all":
        return obs_list
    keep = corner_mask([o.point_id for o in obs_list], mode)
    return [o for o, k in zip(obs_list, keep) if k]


def filter_frames(frames_by_cam: dict[int, list],
                  mode: str) -> dict[int, list]:
    """Apply ``filter_observations`` to every frame across every camera.

    Returns a new dict of ``FrameObservations`` with filtered ``.observations``
    (preserving frame identity, timestamp, image size). Frames that end up
    with zero observations are kept — the pipeline is responsible for
    dropping frames with too few points.
    """
    from mvis.core.types import FrameObservations

    if mode == "all":
        return frames_by_cam

    out: dict[int, list] = {}
    for cid, frames in frames_by_cam.items():
        new_frames = []
        for f in frames:
            kept = filter_observations(f.observations, mode)
            new_frames.append(FrameObservations(
                camera_id=f.camera_id,
                timestamp=f.timestamp,
                observations=kept,
                image_size=f.image_size,
            ))
        out[cid] = new_frames
    return out


def upgrade_high_quality_frames(
    subsampled_frames: dict[int, list],
    full_frames: dict[int, list],
    mode: str,
    num_tags: int,
    threshold_excellent: float = 1.0,
    threshold_good: float = 0.95,
) -> dict[int, list]:
    """Upgrade high-quality frames with more corners.

    Enhance good frames, don't rescue bad ones. Frames with high
    detection rates have reliable corners — giving them more corners
    improves accuracy without adding noise.

      - tags == 100% of board → upgrade to all corners (4/tag)
      - tags >= 95% of board → upgrade to two_diag (2/tag)
      - tags < 95% of board  → keep subsampled (1/tag)

    Args:
        subsampled_frames: Already subsampled frames (e.g. top_left).
        full_frames: Original full-corner frames (before subsampling).
        mode: The subsample mode that was applied (e.g. "top_left").
        num_tags: Total number of tags on the board.
        threshold_excellent: Detection rate for all-corner upgrade (0.85).
        threshold_good: Detection rate for two_diag upgrade (0.70).

    Returns:
        New dict with upgraded high-quality frames.
    """
    import logging
    from mvis.core.types import FrameObservations
    logger = logging.getLogger(__name__)

    if mode == "all":
        return subsampled_frames

    min_tags_excellent = int(num_tags * threshold_excellent)
    min_tags_good = int(num_tags * threshold_good)

    n_upgraded_all = 0
    n_upgraded_diag = 0

    out: dict[int, list] = {}
    for cid, frames in subsampled_frames.items():
        full = full_frames.get(cid, frames)
        full_by_ts = {f.timestamp: f for f in full}

        new_frames = []
        for f in frames:
            n_tags = len({o.point_id // 4 for o in f.observations})

            if n_tags >= min_tags_excellent:
                # Excellent detection — upgrade to all corners
                full_f = full_by_ts.get(f.timestamp)
                if full_f:
                    new_frames.append(full_f)
                    n_upgraded_all += 1
                else:
                    new_frames.append(f)
            elif n_tags >= min_tags_good:
                # Good detection — upgrade to two_diag
                full_f = full_by_ts.get(f.timestamp)
                if full_f:
                    two_diag_obs = [o for o in full_f.observations
                                    if (o.point_id % 4) in (0, 2)]
                    new_frames.append(FrameObservations(
                        camera_id=f.camera_id,
                        timestamp=f.timestamp,
                        observations=two_diag_obs,
                        image_size=f.image_size,
                    ))
                    n_upgraded_diag += 1
                else:
                    new_frames.append(f)
            else:
                # Normal/sparse — keep subsampled (don't amplify noise)
                new_frames.append(f)
        out[cid] = new_frames

    if n_upgraded_all or n_upgraded_diag:
        logger.info(
            f"High-quality upgrade: {n_upgraded_all} frames → all corners, "
            f"{n_upgraded_diag} frames → two_diag")

    return out
