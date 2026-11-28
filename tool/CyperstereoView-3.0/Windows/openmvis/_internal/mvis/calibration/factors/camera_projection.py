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
"""Shared camera projection used by the unified factor and the analysis/report.

This is the single source of truth for the forward projection model. Both
`_make_unified_factor` (in optimizer.py) and the HTML report's error
computation (in generate_report.py) call into this module so there is
no drift between what the optimizer minimizes and what the report shows.

The model applied to every observation:
    1. Time correction: t_corr = t_meas + td + (row / H) * tr
    2. Interpolation:   T_board = SLERP(bp0, bp1, lam(t_corr))
    3. Extrinsic:       T_cam   = T_board.compose(T_ext)   (non-ref cameras)
                        T_cam   = T_board                  (ref camera)
    4. Projection:      pixel   = CameraModel(cal).project(T_cam⁻¹ · point)
                        through `factors.projection._Projector`, the same path
                        for every model (an unregistered model raises)
"""

from typing import Optional

import numpy as np
import gtsam

from mvis.calibration.factors.pose_interpolation import se3_interpolate as _se3_interpolate
from mvis.calibration.factors.projection import make_projector
from mvis.core.camera_models import fit_dist, legacy_model, n_dist


def k_cal_vec(model: str, intr, dist) -> np.ndarray:
    """The camera-only K layout [fx, fy, cx, cy, dist...] of dim
    4 + max(4, n_dist(model)) — 8 for every <=4-coeff model (zero-padded, as
    the old fixed Vector8), 4 + n_dist otherwise. `dist` goes through
    `fit_dist`, so non-zero extras raise (radtan alone keeps its legacy cut to
    4, with a warning). Shared by the optimizer K variable and every residual
    consumer (outlier rejection, report) so they all project the same vector."""
    n = max(4, n_dist(model))
    d = fit_dist(model, np.asarray(dist, float).ravel())
    return np.array([intr[0], intr[1], intr[2], intr[3]] + d + [0.0] * (n - len(d)),
                    dtype=np.float64)


def synced_time(t: float, bp_timeline_t: np.ndarray, tol: float) -> float:
    """The time an observation is projected at: its own stamp, or, when `tol` > 0
    and a ref-camera pose lies within `tol` seconds, exactly that pose's time.

    Without an estimated time offset, a synced camera whose stamps jitter by a few
    ms against the reference would otherwise be interpolated part-way to the next
    ref pose, 100s of ms away at calibration frame rates."""
    if tol > 0 and len(bp_timeline_t):
        k = int(np.argmin(np.abs(bp_timeline_t - t)))
        if abs(bp_timeline_t[k] - t) <= tol:
            return float(bp_timeline_t[k])
    return t


def find_bracketing_poses(
    t: float,
    bp_timeline_t: np.ndarray,
    bp_timeline_idx: list[int],
) -> Optional[tuple[int, int, float, float]]:
    """Find the two nearest reference-camera poses bracketing time `t`.

    Args:
        t: Observation timestamp.
        bp_timeline_t: Sorted array of ref-camera pose timestamps.
        bp_timeline_idx: Parallel list mapping timeline index -> board-pose
            variable index (i.e. the `i` in gtsam.symbol("B", i)).

    Returns:
        (bp0_idx, bp1_idx, t0, t1) with t0 <= t <= t1 when possible,
        or None if `t` is outside the timeline window.
    """
    n = len(bp_timeline_t)
    if n == 0 or t < bp_timeline_t[0] or t > bp_timeline_t[-1]:
        return None

    idx_r = int(np.searchsorted(bp_timeline_t, t))

    if idx_r >= n:
        # Past the end
        idx_l = n - 2
        idx_r = n - 1
    elif idx_r == 0:
        idx_l = 0
        idx_r = min(1, n - 1)
    elif abs(bp_timeline_t[idx_r] - t) < 1e-9:
        # Exact match — use this and its neighbor
        idx_l = idx_r
        if idx_r < n - 1:
            idx_r = idx_r + 1
        else:
            idx_l = idx_r - 1
    else:
        idx_l = idx_r - 1

    bp0_idx = bp_timeline_idx[idx_l]
    bp1_idx = bp_timeline_idx[idx_r]
    t0 = float(bp_timeline_t[idx_l])
    t1 = float(bp_timeline_t[idx_r])
    return bp0_idx, bp1_idx, t0, t1


def _resolve_distortion_model(
    is_fisheye: Optional[bool] = None,
    distortion_model: str = "",
    cal_vec=None,
) -> str:
    """Resolve the distortion model from either is_fisheye or distortion_model
    (the legacy flag only covers 8-long radtan/equidistant cal_vecs)."""
    if distortion_model:
        return distortion_model
    return legacy_model(bool(is_fisheye), cal_vec)


def observation_pose(
    *,
    bp0: gtsam.Pose3,
    bp1: gtsam.Pose3,
    time_meas: float,
    time_0: float,
    time_1: float,
    T_ext: Optional[gtsam.Pose3] = None,
    td: float = 0.0,
    tr: float = 0.0,
    pixel_row: float = 0.0,
    image_height: float = 1.0,
) -> gtsam.Pose3:
    """T_board_cam of one observation: the bracketing poses interpolated at its
    corrected time, then the extrinsic. See `project_observation`."""
    h = max(float(image_height), 1.0)
    t_corr = float(time_meas) + float(td) + (float(pixel_row) / h) * float(tr)
    dt_interval = float(time_1) - float(time_0)
    if abs(dt_interval) < 1e-12:
        lam = 0.0
    else:
        lam = (t_corr - float(time_0)) / dt_interval

    T_board = _se3_interpolate(bp0, bp1, lam)
    return T_board.compose(T_ext) if T_ext is not None else T_board


def project_observation(
    *,
    board_point: np.ndarray,
    bp0: gtsam.Pose3,
    bp1: gtsam.Pose3,
    time_meas: float,
    time_0: float,
    time_1: float,
    cal_vec: np.ndarray,
    is_fisheye: Optional[bool] = None,
    distortion_model: str = "",
    T_ext: Optional[gtsam.Pose3] = None,
    td: float = 0.0,
    tr: float = 0.0,
    pixel_row: float = 0.0,
    image_height: float = 1.0,
) -> Optional[np.ndarray]:
    """Project a board point to pixel coordinates.

    Identical forward model used by `_make_unified_factor` and by the
    analysis/report — if both call this function, their residuals agree
    by construction.

    Args:
        board_point: (3,) 3D point in the board frame.
        bp0, bp1: Bracketing reference-camera poses (T_board_cam_ref) at
            timestamps `time_0` and `time_1`.
        time_meas: Observation timestamp.
        time_0, time_1: Bracketing pose timestamps.
        cal_vec: intrinsics [fx, fy, cx, cy, dist...] (8 for radtan /
            equidistant; >= 4 + n_dist for custom models).
        is_fisheye: DEPRECATED — use distortion_model instead. If True use
            Cal3Fisheye (equidistant), else Cal3DS2 (radtan).
        distortion_model: a `core.camera_models.MODELS` name. Takes precedence
            over is_fisheye.
        T_ext: Optional extrinsic T_ref_cam.
        td: Time offset (seconds).
        tr: Rolling-shutter readout (seconds).
        pixel_row: Row of the observation (for rolling shutter).
        image_height: Image height in pixels.

    Returns:
        (2,) projected pixel, or None if projection fails.
    """
    model = _resolve_distortion_model(is_fisheye, distortion_model, cal_vec)

    T_cam = observation_pose(bp0=bp0, bp1=bp1, time_meas=time_meas, time_0=time_0, time_1=time_1,
                             T_ext=T_ext, td=td, tr=tr, pixel_row=pixel_row,
                             image_height=image_height)
    return project_at_pose(T_cam, board_point, cal_vec, model)


def project_at_pose(T_cam: gtsam.Pose3, board_point: np.ndarray, cal_vec: np.ndarray,
                    model: str) -> Optional[np.ndarray]:
    """(2,) pixel of a board point seen from T_cam, or None if projection fails
    (behind the camera / out of the model's range). Every model goes through the
    one `_Projector` (CameraModel) path the factors use."""
    try:
        return make_projector(model, cal_vec).project(T_cam, board_point, jac=False)[0]
    except RuntimeError:
        return None
