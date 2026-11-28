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
"""Synthetic camera measurements from a BsplineSE3 trajectory (simulator Phase 3).

Projects a known scene (calibration-board corners by default) into each camera
frame via the camera intrinsics + camera-IMU extrinsic ``T_CtoI`` + time offset
``td``. Global shutter (Phase 3a); rolling shutter is added in 3b.

The camera pose in global is ``T_CtoG = T_ItoG(t + td) · T_CtoI`` and projection
reuses gtsam's PinholeCamera (radtan ``Cal3DS2`` / fisheye ``Cal3Fisheye``), so
the generated ``(feat_id, uv)`` per frame match what the calibration pipeline
already consumes. Feature correspondences are known by construction (no tracker).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import gtsam


def board_points_in_global(board, T_BinG: gtsam.Pose3) -> np.ndarray:
    """World-frame 3-D corners of a `board/` target placed at pose T_BinG."""
    pts_B = np.asarray(board.all_points_3d(), float)
    return np.array([T_BinG.transformFrom(gtsam.Point3(*p)) for p in pts_B])


def place_board_facing_camera(traj, t0: float, depth: float, board,
                              T_CtoI: gtsam.Pose3 | None = None) -> gtsam.Pose3:
    """Auto-place a board so it faces the camera at trajectory time ``t0``.

    The board centroid is put ``depth`` metres in front of the camera along its
    optical axis, and the board is oriented so its surface normal points back at
    the camera (and stays axis-aligned with the image). Returns ``T_BinG``; feed
    it to :func:`board_points_in_global`. Guarantees the board is centred in view
    at ``t0`` (with calibration-waving motion it stays visible).
    """
    T_CtoI = gtsam.Pose3() if T_CtoI is None else T_CtoI
    R_GtoI, p_IinG = traj.get_pose(t0)
    T_CtoG = gtsam.Pose3(gtsam.Rot3(R_GtoI.T), gtsam.Point3(*p_IinG)).compose(T_CtoI)
    R_CtoG = T_CtoG.rotation().matrix()
    x_cam, y_cam, z_cam = R_CtoG[:, 0], R_CtoG[:, 1], R_CtoG[:, 2]   # optical axis = z_cam
    center = np.asarray(T_CtoG.translation(), float) + depth * z_cam

    # board frame: normal (+z_B) points back at the camera; x_B along camera x
    R_BinG = np.column_stack([x_cam, -y_cam, -z_cam])               # proper rotation
    centroid_B = board.all_points_3d().mean(0)
    t_BinG = center - R_BinG @ centroid_B
    return gtsam.Pose3(gtsam.Rot3(R_BinG), gtsam.Point3(*t_BinG))


def project_points(T_CtoG: gtsam.Pose3, cal, points_inG, image_size,
                   z_near: float = 0.05, z_far: float = 15.0):
    """Project world points into a camera at pose T_CtoG. Returns (ids, uv) of the
    points in front of the camera (z in [z_near, z_far]) and inside the image."""
    w, h = image_size
    ids, uv = [], []
    for i, P in enumerate(np.asarray(points_inG, float)):
        pc = np.asarray(T_CtoG.transformTo(gtsam.Point3(*P)), float)
        z = pc[2]
        if z < z_near or z > z_far:
            continue
        try:
            px = np.asarray(cal.uncalibrate(np.array([pc[0] / z, pc[1] / z])), float)
        except RuntimeError:
            continue
        if 0.0 <= px[0] < w and 0.0 <= px[1] < h:
            ids.append(i)
            uv.append(px)
    return np.array(ids, int), np.array(uv, float).reshape(-1, 2)


def _rs_row_time(t0: float, row: float, h: int, readout: float, mode: int) -> float:
    """Capture time of image `row` for a rolling-shutter frame referenced at t0.

    readout = total top-to-bottom readout time (seconds). Modes (match MVIS):
      0: frame time = top of frame      -> t0 + (row/H)·tr
      1: frame time = middle of frame   -> t0 + (row/H - 0.5)·tr
      2: frame time = bottom of frame   -> t0 + (row - H)/H·tr
    """
    if mode == 0:
        return t0 + (row / h) * readout
    if mode == 1:
        return t0 + (row / h - 0.5) * readout
    return t0 + (row - h) / h * readout


def project_points_rolling(traj, t0: float, cal, T_CtoI: gtsam.Pose3, points_inG,
                           image_size, readout: float, rolling_mode: int = 0,
                           z_near: float = 0.05, z_far: float = 15.0,
                           max_iter: int = 10, row_tol: float = 0.05):
    """Rolling-shutter projection: each feature's capture time depends on the row
    it lands on, solved by fixed-point iteration. `t0` is the frame's reference
    trajectory time. Returns (ids, uv, capture_times)."""
    w, h = image_size
    ids, uv_out, t_out = [], [], []
    for i, P in enumerate(np.asarray(points_inG, float)):
        Ppt = gtsam.Point3(*P)
        ti = t0
        prev_row, last_uv, good = -1.0, None, False
        for _ in range(max_iter):
            R_GtoI, p = traj.get_pose(ti)
            T_CtoG = gtsam.Pose3(gtsam.Rot3(R_GtoI.T), gtsam.Point3(*p)).compose(T_CtoI)
            pc = np.asarray(T_CtoG.transformTo(Ppt), float)
            z = pc[2]
            if z < z_near or z > z_far:
                good = False
                break
            try:
                px = np.asarray(cal.uncalibrate(np.array([pc[0] / z, pc[1] / z])), float)
            except RuntimeError:
                good = False
                break
            if not (0.0 <= px[0] < w and 0.0 <= px[1] < h):
                good = False
                break
            last_uv, good = px, True
            row = px[1]
            ti = _rs_row_time(t0, row, h, readout, rolling_mode)
            if abs(row - prev_row) < row_tol:
                break
            prev_row = row
        if good and last_uv is not None:
            ids.append(i)
            uv_out.append(last_uv)
            t_out.append(ti)
    return (np.array(ids, int), np.array(uv_out, float).reshape(-1, 2),
            np.array(t_out, float))


@dataclass
class CamStream:
    """Per-frame camera detections (known feature IDs + pixels)."""
    t: np.ndarray            # (F,) frame timestamps
    ids: list                # per frame: int array of visible feature IDs
    uv: list                 # per frame: (n_i, 2) pixel array


def simulate_camera(traj, times, cal, T_CtoI: gtsam.Pose3, points_inG, image_size,
                    td: float = 0.0, sigma_px: float = 0.0, readout: float = 0.0,
                    rolling_mode: int = 0,
                    z_near: float = 0.05, z_far: float = 15.0, seed: int = 0):
    """Generate a `CamStream` by projecting `points_inG` into the camera at each
    frame time. The camera pose is `T_ItoG(t + td) · T_CtoI`; `cal` is a gtsam
    Cal3DS2/Cal3Fisheye; `sigma_px` adds Gaussian pixel noise (0 = noise-free).

    Non-zero `readout` enables rolling shutter (per-row capture time, `rolling_mode`
    0/1/2; negative = bottom row first); `readout` = 0 is global shutter.
    """
    times = np.asarray(times, float)
    rng = np.random.RandomState(seed)
    out_ids, out_uv = [], []
    for t in times:
        t0 = t + td
        if readout != 0.0:
            ids, uv, _ts = project_points_rolling(
                traj, t0, cal, T_CtoI, points_inG, image_size, readout,
                rolling_mode, z_near, z_far)
        else:
            R_GtoI, p = traj.get_pose(t0)
            T_CtoG = gtsam.Pose3(gtsam.Rot3(R_GtoI.T), gtsam.Point3(*p)).compose(T_CtoI)
            ids, uv = project_points(T_CtoG, cal, points_inG, image_size, z_near, z_far)
        if sigma_px and len(uv):
            uv = uv + rng.standard_normal(uv.shape) * sigma_px
        out_ids.append(ids)
        out_uv.append(uv)
    return CamStream(times, out_ids, out_uv)
