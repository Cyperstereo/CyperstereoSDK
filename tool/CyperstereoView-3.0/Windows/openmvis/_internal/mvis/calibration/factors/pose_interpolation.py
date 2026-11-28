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
"""Pose interpolation / propagation — operations + their Jacobians, in one place.

Two distinct concerns, both kept here (op and Jacobian paired) so the
calibration factors share one home for pose-time math:

1. **Camera rolling-shutter SLERP** — interpolate a board pose *between* two
   bracketing poses by λ∈[0,1]:
     - `se3_interpolate(B0, B1, λ)`            (decoupled: rotation SLERP + linear translation)
     - `se3_interpolate_jacobians(B0, B1, λ)`  (the same decoupled model + its Jacobians)
   and, for the IMU-state interpolation of the cam-IMU vision factors
   (`CameraCalibInterpFactor`, `CameraFeatureInterpFactor`):
     - `interp_pose(X_k, X_k1, λ)`              (R_k Exp(λφ), (1-λ) p_k + λ p_k1)
     - `interp_pose_tangents(φ, A, R_λ, λ, dp)` (its right-tangent state blocks)

2. **IMU time-offset propagation** — shift a single base-IMU pose forward by the
   time offset td (using angular velocity) and apply the aux↔base extrinsic, to
   get the auxiliary-IMU pose. Shared by the multi-IMU and gyro factors:
     - `imu_aux_rotation(...)`, `imu_aux_pose(...)`        (op)
     - `imu_aux_pose_jacobians(...)`                        (Jacobian: par_p, par_v, par_calib)
"""

from __future__ import annotations

import numpy as np
import gtsam

from mvis.calibration.factors.lie_groups import (
    skew, exp_so3, log_so3, so3_right_jacobian,
    so3_right_jacobian_inv, so3_left_jacobian_inv,
)


# ---------------------------------------------------------------------------
# 1. Camera rolling-shutter SLERP (between two poses)
# ---------------------------------------------------------------------------
def se3_interpolate(pose0: gtsam.Pose3, pose1: gtsam.Pose3, lam: float) -> gtsam.Pose3:
    """Interpolate between two SE(3) poses: rotation SLERP + linear translation,
    i.e. `interp_pose` with the exact end poses at lam = 0 and 1.

    rotation: R0 · Exp(λ · Log(R0^T R1)); translation: (1-λ) p0 + λ p1.
    λ is NOT clamped (extrapolation allowed, so dλ Jacobians stay non-zero at
    the boundaries).
    """
    end = _end_pose(pose0, pose1, lam)
    return end if end is not None else interp_pose(pose0, pose1, lam)[0]


def _end_pose(pose0, pose1, lam):
    """pose0 / pose1 when lam is within 1e-10 of 0 / 1, else None."""
    if abs(lam) < 1e-10:
        return pose0
    if abs(lam - 1.0) < 1e-10:
        return pose1
    return None


def se3_interpolate_jacobians(B0: gtsam.Pose3, B1: gtsam.Pose3, lam: float):
    """`se3_interpolate` bundled with its analytic Jacobians (`interp_pose_tangents`).

    Returns (T_board, J_B0 (6×6), J_B1 (6×6), v_lam (6,)) in gtsam Pose3 tangent
    [rot; trans]. The translation/rotation cross-blocks are zero (decoupled model).
    """
    T, phi, A, R_lam = interp_pose(B0, B1, lam)
    dp = np.asarray(B1.translation(), float) - np.asarray(B0.translation(), float)
    J_B0, J_B1, v_lam = interp_pose_tangents(phi, A, R_lam, lam, dp)
    end = _end_pose(B0, B1, lam)
    return (end if end is not None else T), J_B0, J_B1, v_lam


def interp_pose(pose_k: gtsam.Pose3, pose_k1: gtsam.Pose3, lam: float):
    """T(lam) between two IMU states, plus the pieces the Jacobians reuse.

    R(lam) = R_k Exp(lam phi), phi = Log(R_k^T R_k1); p(lam) = (1 - lam) p_k + lam p_k1.
    Returns (T(lam), phi, A = Exp(lam phi), R(lam)). `se3_interpolate` is this
    pose with exact end poses.
    """
    R_k = pose_k.rotation().matrix()
    p_k = np.asarray(pose_k.translation(), float)
    p_k1 = np.asarray(pose_k1.translation(), float)
    phi = log_so3(R_k.T @ pose_k1.rotation().matrix())
    A = exp_so3(lam * phi)
    R_lam = R_k @ A
    p_lam = (1.0 - lam) * p_k + lam * p_k1
    return gtsam.Pose3(gtsam.Rot3(R_lam), gtsam.Point3(*p_lam)), phi, A, R_lam


def interp_pose_tangents(phi, A, R_lam, lam: float, dp):
    """Right-tangent blocks of `interp_pose` w.r.t. the two states and lam.

    `phi, A, R_lam` come from `interp_pose`; dp = p_k1 - p_k. Returns
    (M_k 6x6, M_k1 6x6, zeta_lam 6): d zeta / d X_k, d zeta / d X_k1, d zeta / d lam.
    """
    Jr_lp = so3_right_jacobian(lam * phi)
    M_k = np.zeros((6, 6))
    M_k[0:3, 0:3] = A.T - lam * Jr_lp @ so3_left_jacobian_inv(phi)
    M_k[3:6, 3:6] = (1.0 - lam) * A.T
    M_k1 = np.zeros((6, 6))
    M_k1[0:3, 0:3] = lam * Jr_lp @ so3_right_jacobian_inv(phi)
    M_k1[3:6, 3:6] = lam * exp_so3((1.0 - lam) * phi)
    zeta_lam = np.concatenate([phi, R_lam.T @ dp])
    return M_k, M_k1, zeta_lam


# ---------------------------------------------------------------------------
# 2. IMU time-offset pose propagation (single pose shifted by td)
# ---------------------------------------------------------------------------
# Naming: I = base IMU, Is = auxiliary (sensor) IMU; R_IstoI / p_IsinI = extrinsic;
# td = aux time offset; w_inIs = corrected aux angular velocity (interval endpoint).
def imu_aux_rotation(base_pose, R_IstoI, td, w_inIs):
    """Aux-IMU orientation in global: R_base · Exp(R_IstoI·w_inIs·td) · R_IstoI."""
    R_base = base_pose.rotation().matrix()
    return (R_base @ exp_so3(R_IstoI @ w_inIs * td)) @ R_IstoI


def imu_aux_pose(base_pose, R_IstoI, p_IsinI, td, w_inIs, v_base):
    """Aux-IMU pose from the base pose via extrinsic + time-offset propagation."""
    R_base = base_pose.rotation().matrix()
    R_ItoG = R_base @ exp_so3(R_IstoI @ w_inIs * td)
    R_IstoG = R_ItoG @ R_IstoI
    p_IinG = np.asarray(base_pose.translation(), float) + np.asarray(v_base, float) * td
    p_IsinG = p_IinG + R_ItoG @ np.asarray(p_IsinI, float)
    return gtsam.Pose3(gtsam.Rot3(R_IstoG), gtsam.Point3(*p_IsinG))


def imu_aux_pose_jacobians(base_pose, R_IstoI, p_IsinI, td, w_inIs, v_base):
    """Jacobians of the aux-IMU pose tangent w.r.t. base pose / base velocity /
    [calib.T tangent(6), td(1)] — (par_p 6×6, par_v 6×3, par_calib 6×7).

    The rotation rows (0:3) are what the gyro factor uses; the full 6 rows the
    multi-IMU factor. (Ports the C++ par_ps_*; par_v uses R_IstoG^T·td — the
    body-frame rotation — not the C++ I·td, see the IMU Jacobian notes.)
    """
    R_pose = base_pose.rotation().matrix()
    dRs = exp_so3(w_inIs * td)
    dth = R_IstoI @ w_inIs * td
    Exp_dth = exp_so3(dth)                       # R_ItoG = R_pose @ Exp_dth
    R_IstoG = R_pose @ Exp_dth @ R_IstoI
    RgT = R_IstoG.T
    p_IsinI = np.asarray(p_IsinI, float)
    v_base = np.asarray(v_base, float)

    par_p = np.zeros((6, 6))
    par_p[0:3, 0:3] = (R_IstoI @ dRs).T
    par_p[3:6, 0:3] = -RgT @ R_pose @ skew(Exp_dth @ p_IsinI)
    par_p[3:6, 3:6] = RgT @ R_pose

    par_v = np.zeros((6, 3))
    par_v[3:6, 0:3] = RgT * td                   # body-frame; not I·td (see notes)

    par_calib = np.zeros((6, 7))
    par_calib[0:3, 0:3] = dRs.T
    par_calib[0:3, 6] = w_inIs
    par_calib[3:6, 0:3] = RgT @ R_pose @ Exp_dth @ skew(p_IsinI) @ so3_right_jacobian(dth) @ skew(w_inIs * td)
    par_calib[3:6, 3:6] = RgT @ R_pose @ Exp_dth @ R_IstoI
    par_calib[3:6, 6] = RgT @ (v_base + R_pose @ Exp_dth @ skew(R_IstoI @ w_inIs) @ p_IsinI)
    return par_p, par_v, par_calib
