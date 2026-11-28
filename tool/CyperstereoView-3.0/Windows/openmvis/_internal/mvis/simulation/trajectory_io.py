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
"""Simulator trajectory I/O (Phase 2d).

Canonical in-repo trajectory format (one pose per row):

    t  theta_x theta_y theta_z  px py pz

where ``theta_ItoG = Log(R_ItoG)`` is the rotation-vector of the body->global
rotation and ``p = p_IinG``. This replaces the MVIS row format
``[t, p_IinG, q_GtoI]`` (JPL quaternion, convention-ambiguous): a rotation-vector
is self-describing and matches our gtsam ``Exp/Log`` convention, so the loader is
convention-safe (``R_ItoG = Exp(theta)`` is exactly orthonormal — no quaternion
normalization or determinant drift).

`convert_mvis_trajectory` produces this format from an MVIS file; `load_trajectory`
reads it into a `BsplineSE3` (resampling to uniform control-point spacing by
SE(3) interpolation, as BsplineSE3Wheels::feed_trajectory does).
"""

from __future__ import annotations

import numpy as np
import gtsam

from mvis.calibration.factors.lie_groups import skew
from mvis.simulation.bspline_se3 import BsplineSE3


def _quat_jpl_to_R_GtoI(q) -> np.ndarray:
    """JPL quaternion [qx,qy,qz,qw] -> R_GtoI (matches ov_core::quat_2_Rot)."""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q)
    qv, qw = q[0:3], float(q[3])
    return (2 * qw * qw - 1) * np.eye(3) - 2 * qw * skew(qv) + 2 * np.outer(qv, qv)


def _interp_se3(P0: np.ndarray, P1: np.ndarray, lam: float) -> np.ndarray:
    """SE(3) interpolation exp(lam·log(P1·P0^-1))·P0 (feed_trajectory convention)."""
    rel = gtsam.Pose3(P1).compose(gtsam.Pose3(P0).inverse())
    step = gtsam.Pose3.Expmap(lam * gtsam.Pose3.Logmap(rel))
    return step.compose(gtsam.Pose3(P0)).matrix()


def convert_mvis_trajectory(in_path: str, out_path: str, header: str = "") -> int:
    """Convert an MVIS trajectory (t, p_IinG, q_GtoI[JPL]) to the canonical
    format (t, theta_ItoG, p_IinG). Returns the number of rows written."""
    data = np.loadtxt(in_path)
    if data.ndim != 2 or data.shape[1] < 8:
        raise ValueError(f"{in_path}: expected >=8 cols (t,p,q), got {data.shape}")
    t, p, q = data[:, 0], data[:, 1:4], data[:, 4:8]
    out = np.zeros((len(t), 7))
    out[:, 0] = t
    for i in range(len(t)):
        R_ItoG = _quat_jpl_to_R_GtoI(q[i]).T
        out[i, 1:4] = gtsam.Rot3.Logmap(gtsam.Rot3(R_ItoG))
        out[i, 4:7] = p[i]
    hdr = (header + "\n" if header else "") + "t theta_ItoG(3) p_IinG(3)"
    np.savetxt(out_path, out, fmt="%.9f", header=hdr)
    return len(out)


def load_trajectory(path: str, dt: float | None = None) -> BsplineSE3:
    """Read the canonical 7-col trajectory (t, theta_ItoG, p_IinG) into a
    BsplineSE3, resampling to uniform control spacing ``dt`` (default: median
    input timestep)."""
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 7:
        raise ValueError(f"expected >=7 columns (t,theta,p), got shape {data.shape}")
    t = data[:, 0]
    theta = data[:, 1:4]
    p = data[:, 4:7]

    poses = []
    for i in range(len(t)):
        T = np.eye(4)
        T[0:3, 0:3] = gtsam.Rot3.Expmap(theta[i]).matrix()    # R_ItoG (exact SO(3))
        T[0:3, 3] = p[i]
        poses.append(T)

    if dt is None:
        dt = float(np.median(np.diff(t)))
    n = int(np.floor((t[-1] - t[0]) / dt))
    ctrl_times = t[0] + np.arange(n + 1) * dt
    idx = np.clip(np.searchsorted(t, ctrl_times, side="right") - 1, 0, len(t) - 2)

    ctrl = []
    for j, tc in zip(idx, ctrl_times):
        span = t[j + 1] - t[j]
        lam = 0.0 if span <= 0 else min(max((tc - t[j]) / span, 0.0), 1.0)
        ctrl.append(_interp_se3(poses[j], poses[j + 1], lam))
    return BsplineSE3(float(t[0]), float(dt), ctrl)
