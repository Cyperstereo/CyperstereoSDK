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
"""Two-phase camera-only solve schedule for one double_sphere or omni_radtan
camera (`solve_phased`).

omni_radtan:
  P1  k1, k2, p1, p2 held at the seed, xi under a sigma 1.0 prior at the seed,
      fx, fy, cx, cy and the board poses free;
  P2  xi held at the P1 value, everything else free (k1 of the Mei model and
      xi trade off, PMC11510980).

A single-camera fit therefore keeps xi at the P1 value unless pipeline Stage
2's diverged-pose reopt runs (that solve has xi free); a rig fits xi free in
Stage 4. A final free-xi stage was tried for #26 and dropped: with corners
below about 80 deg single-camera xi is weakly pinned, and on T265 cam1/07 a
free xi slid from 1.919 to 1.635 (Kalibr 1.900) with fx 835 -> 754 and k1
0.170 -> 0.003 at a 0.0025 px lower RMS.

double_sphere:
  P1  xi, alpha held at the seed (the canonical Usenko init), fx, fy, cx, cy
      and the board poses free;
  P2  everything free, warm-started from P1. xi / alpha can absorb fx; with
      no P1 anchor the solve escapes to a basin where xi flips positive and fx
      inflates about 2x (vs Usenko 2018 Table 3; found 2026-05-01).

P2 runs with the P1 optimizer's own options; omni_radtan only adds its camera
to `frozen_xi_cameras` (`phase2_options`). Earlier, pipeline Stage 2 rebuilt
the omni P2 options by hand and dropped `board_heights` / `board_height_sigma`,
so a board-shape solve lost the board shape in P2.

Used by pipeline Stage 2 (so `mvis cam` / `mvis run`) and by `mvis calibrate` (every camera of
a phased model, before its solve). Earlier, the CLI ran the omni
schedule only and solved double_sphere in one go.
"""
from __future__ import annotations

import dataclasses
import logging

logger = logging.getLogger("mvis.calibration.omni_schedule")

PHASED_MODELS = frozenset({"double_sphere", "omni_radtan"})


def poses_by_frame(result, n_frames, fallback):
    """Board poses of `result` re-indexed to the frame list (None where the
    solve had no pose variable)."""
    poses = [None] * n_frames
    vfi = result.valid_frame_indices
    if not vfi:
        vfi = [i for i, p in enumerate(fallback) if p is not None]
    if result.board_poses_gtsam:
        for fi, pose in zip(vfi, result.board_poses_gtsam):
            if fi < n_frames:
                poses[fi] = pose
    return poses


def phase2_options(options, distortion_model, camera_id):
    """P2 options: `options` (the P1 optimizer's) with, for omni_radtan,
    `camera_id` added to `frozen_xi_cameras`; double_sphere returns `options`
    itself (P2 only lifts frozen_distortion)."""
    if distortion_model == "omni_radtan":
        frozen = set(options.frozen_xi_cameras or ()) | {camera_id}
        return dataclasses.replace(options, frozen_xi_cameras=frozen)
    if distortion_model == "double_sphere":
        return options
    raise ValueError(f"no two-phase schedule for {distortion_model!r}")


def solve_phased(optimizer, frames, intrinsics, dist_coeffs, init_poses,
                 distortion_model, image_size, camera_id):
    """P1 -> P2 for one double_sphere or omni_radtan camera; returns the P2
    result. P1 runs on `optimizer` with frozen_distortion; P2 on an optimizer
    of the same board and `phase2_options(optimizer.options, ...)`."""
    cid, dm = camera_id, distortion_model
    tag = "DS" if dm == "double_sphere" else "omni"
    opts2 = phase2_options(optimizer.options, dm, cid)
    optimizer2 = (optimizer if opts2 is optimizer.options
                  else type(optimizer)(optimizer.board, opts2))
    if dm == "double_sphere":
        logger.info(f"  Camera {cid}: DS phase 1 (distortion frozen at "
                    f"xi={dist_coeffs[0]:.3f}, alpha={dist_coeffs[1]:.3f})")
    else:
        logger.info(f"  Camera {cid}: omni phase 1 (distortion frozen)...")
    r1 = optimizer.solve_single_camera(
        frames, intrinsics, dist_coeffs, init_poses, dm, image_size,
        camera_id=cid, frozen_distortion=True)
    c1 = r1.cameras[0]
    logger.info(f"  Camera {cid}: {tag} phase 1 → fx={c1.fx:.2f} "
                f"xi={c1.distortion[0]:.4f} "
                + (f"alpha={c1.distortion[1]:.4f}" if dm == "double_sphere" else ""))
    logger.info(f"  Camera {cid}: {tag} phase 2 ("
                + ("all params free)" if dm == "double_sphere"
                   else "xi frozen, k1/k2/p1/p2 free)..."))
    p1_poses = poses_by_frame(r1, len(frames), init_poses)
    r2 = optimizer2.solve_single_camera(
        frames, c1.intrinsics, c1.distortion, p1_poses, dm, image_size,
        camera_id=cid)
    c2 = r2.cameras[0]
    logger.info(f"  Camera {cid}: {tag} phase 2 → fx={c2.fx:.2f} "
                + " ".join(f"{n}={v:.4f}" for n, v in zip(
                    ("xi", "alpha") if dm == "double_sphere"
                    else ("xi", "k1", "k2"), c2.distortion)))
    return r2
