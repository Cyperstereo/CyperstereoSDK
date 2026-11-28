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
"""Interpolated-pose camera reprojection factor over the CameraCalib bundle.

The vision factor for the multi-camera graph — and for the BASE camera too: a frame stamped t is constrained by the TWO
bracketing IMU states X_k, X_{k+1} (stamps t_k < t_{k+1}); the IMU pose at
each corner's CAPTURE time is interpolated on SO(3) x R^3:

    lam_i  = (dt_frame + td + rho_i * tr) / dt_interval
    phi    = Log(R_k^T R_{k+1})
    R(lam) = R_k Exp(lam phi)
    p(lam) = (1 - lam) p_k + lam p_{k+1}
    T_CtoW = ( R(lam), p(lam) ) . T_CtoI

with dt_frame = t - t_k and dt_interval = t_{k+1} - t_k held constant. The
time offsets td/tr are estimated THROUGH lam (d/dtd = (1/dt) d/dlam), so no
gyro sample or velocity state enters — vb drops out of vision factors
entirely. Compared to the single-state CameraCalibFactor (Exp(w m td)
propagation; now the test fixture tests/helpers/camera_calib_factor.py),
td sensitivity here comes from the inter-state MEAN motion — the same
interval-mean-beats-instantaneous lesson as the init's td stage.

Analytic Jacobians; right-tangent blocks of T(lam) w.r.t. the states:

    zeta_theta/d theta_k     = A^T - lam Jr(lam phi) Jl^-1(phi)
    zeta_rho  /d rho_k       = (1 - lam) A^T
    zeta_theta/d theta_{k+1} = lam Jr(lam phi) Jr^-1(phi)
    zeta_rho  /d rho_{k+1}   = lam Exp((1-lam) phi)
    d zeta / d lam           = [phi ; R(lam)^T (p_{k+1} - p_k)]

with A = Exp(lam phi). Sanity built in: at lam=0 the X_k block is identity /
X_{k+1} block zero, and vice versa at lam=1 (Jr(phi)Jl^-1(phi) = Exp(-phi)).
All blocks FD-checked (`_numerical_jacobians`, tests at interior lam).
"""

from __future__ import annotations

import numpy as np
import gtsam

from .camera_calib import CameraCalib
from ._calib_memo import get_calib_setup
from .lie_groups import skew
from .pose_interpolation import interp_pose, interp_pose_tangents
from .corner_reproj import CornerReprojMixin
from .numdiff import numerical_jacobian


# Old private name, kept for the tests that import it from here (no src importer).
_interp_pose = interp_pose


class CameraCalibInterpFactor(CornerReprojMixin):
    """Per-frame stacked reprojection over [pose_k, pose_k1, board, calib(ξ16)]."""

    def __init__(self, uv_meas, p_FinB, dt_frame: float, dt_interval: float,
                 calib_base: CameraCalib, image_size, rolling_mode: int = 0,
                 board_pids=None, n_board_points: int = 0):
        self.uv = np.asarray(uv_meas, float).reshape(-1, 2)
        self.p_B = np.asarray(p_FinB, float).reshape(-1, 3)
        assert len(self.uv) == len(self.p_B), "one board corner per measurement"
        assert dt_interval > 0
        # Estimated board shape (optional): `board_pids` are the corners' board
        # point ids; the factor then takes a trailing key holding one height
        # offset per board point (length n_board_points), added along the board
        # normal: p_B_i + [0, 0, h[pid_i]].
        self.pids = None if board_pids is None else np.asarray(board_pids, int).reshape(-1)
        self.n_bp = int(n_board_points)
        if self.pids is not None:
            assert len(self.pids) == len(self.uv) and self.n_bp > int(self.pids.max())
        self.dt_frame = float(dt_frame)
        self.dt = float(dt_interval)
        self.base = calib_base
        self.w_px, self.h_px = int(image_size[0]), int(image_size[1])
        self.rolling_mode = int(rolling_mode)
        self.n = len(self.uv)
        self.rho = self._row_offsets()
        self.keys = None                        # set by make_factor

    # ------------------------------------------------------------------
    @property
    def uses_heights_key(self) -> bool:
        return self.pids is not None

    def error(self, pose_k: gtsam.Pose3, pose_k1: gtsam.Pose3,
              T_BinW: gtsam.Pose3, xi, h=None):
        """Stacked 2N reprojection residual (unwhitened)."""
        return self._residuals(pose_k, pose_k1, T_BinW, xi, h)[0]

    def _residuals(self, pose_k, pose_k1, T_BinW, xi, h=None):
        """(stacked 2N unwhitened residual, per-corner validity mask). Invalid
        corners (cheirality / model-range guard) keep zero rows in the residual
        but are False in the mask so stats can exclude them."""
        setup = get_calib_setup(self.base, xi)      # shared memo (read-only)
        calib, proj = setup.calib, setup.proj
        err = np.zeros(2 * self.n)
        valid = np.zeros(self.n, bool)
        pB = self._pB(h)
        T_I_cache = None                        # shared when tr == 0
        for i in range(self.n):
            if calib.tr == 0.0 and T_I_cache is not None:
                T_I = T_I_cache
            else:
                T_I, _phi, _A, _R = interp_pose(pose_k, pose_k1,
                                                self._lam(calib, i))
                if calib.tr == 0.0:
                    T_I_cache = T_I
            T_CtoW = T_I.compose(calib.T)
            p_W = T_BinW.transformFrom(gtsam.Point3(*pB[i]))
            try:
                px, _, _, _ = proj.project(T_CtoW, p_W, jac=False)
            except RuntimeError:                # cheirality / model range
                continue
            err[2 * i:2 * i + 2] = px - self.uv[i]
            valid[i] = True
        return err, valid

    def camera_points_values(self, values: gtsam.Values) -> np.ndarray:
        """Each corner in the camera frame at a gtsam Values (N x 3), by the same
        chain as `_residuals`; for the corner accounting, which needs the angle."""
        pose_k, pose_k1, T_BinW, xi, h = self.args_from_values(values)
        calib = get_calib_setup(self.base, xi).calib
        pB = self._pB(h)
        out = np.empty((self.n, 3))
        for i in range(self.n):
            T_I = interp_pose(pose_k, pose_k1, self._lam(calib, i))[0]
            p_W = T_BinW.transformFrom(gtsam.Point3(*pB[i]))
            out[i] = np.asarray(T_I.compose(calib.T).transformTo(p_W), float)
        return out

    def args_from_values(self, values: gtsam.Values):
        """(pose_k, pose_k1, board, xi, h|None) at a gtsam Values, using the keys
        from make_factor."""
        k = self.keys
        h = values.atVector(k[4]) if self.uses_heights_key else None
        return (values.atPose3(k[0]), values.atPose3(k[1]),
                values.atPose3(k[2]), values.atVector(k[3]), h)

    # ------------------------------------------------------------------
    def error_and_jacobians(self, pose_k, pose_k1, T_BinW, xi, h=None):
        """(error 2N, [H_pose_k 2N×6, H_pose_k1 2N×6, H_board 2N×6,
        H_calib 2N×(8+n_intr), (H_heights 2N×n_board_points when estimated)])."""
        setup = get_calib_setup(self.base, xi)      # shared memo (read-only)
        calib, proj = setup.calib, setup.proj
        n_intr = calib.n_intr
        R_B = T_BinW.rotation().matrix()
        T_C = calib.T
        Ad_TCinv = setup.Ad_TCinv
        Jr_xiT = setup.Jr_xiT

        p_k = np.asarray(pose_k.translation(), float)
        p_k1 = np.asarray(pose_k1.translation(), float)
        dp = p_k1 - p_k

        err = np.zeros(2 * self.n)
        H_pk = np.zeros((2 * self.n, 6))
        H_pk1 = np.zeros((2 * self.n, 6))
        H_board = np.zeros((2 * self.n, 6))
        H_calib = np.zeros((2 * self.n, calib.DIM))
        H_h = np.zeros((2 * self.n, self.n_bp)) if self.uses_heights_key else None
        pB = self._pB(h)
        n_B = R_B[:, 2]                         # d p_W / d h  (board normal in W)

        cache = {}
        for i in range(self.n):
            lam = self._lam(calib, i)
            if lam in cache:
                T_I, phi, A, R_lam, M_k, M_k1, zeta_lam = cache[lam]
            else:
                T_I, phi, A, R_lam = interp_pose(pose_k, pose_k1, lam)
                M_k, M_k1, zeta_lam = interp_pose_tangents(phi, A, R_lam, lam, dp)
                cache[lam] = (T_I, phi, A, R_lam, M_k, M_k1, zeta_lam)

            T_CtoW = T_I.compose(T_C)
            p_W = np.asarray(T_BinW.transformFrom(gtsam.Point3(*pB[i])), float)

            try:
                px, J_pose, J_point, J_intr = proj.project(T_CtoW, p_W)
            except RuntimeError:                # cheirality: zero rows
                continue
            err[2 * i:2 * i + 2] = px - self.uv[i]
            r = slice(2 * i, 2 * i + 2)
            J_T = J_pose @ Ad_TCinv            # w.r.t. T(lam) right tangent

            H_pk[r] = J_T @ M_k
            H_pk1[r] = J_T @ M_k1
            H_board[r, 0:3] = J_point @ (-R_B @ skew(pB[i]))
            H_board[r, 3:6] = J_point @ R_B
            if H_h is not None:
                H_h[r, self.pids[i]] = J_point @ n_B
            H_calib[r, 0:6] = J_pose @ Jr_xiT
            col_t = J_T @ zeta_lam / self.dt   # d/dtd through lam
            H_calib[r, 6] = col_t
            H_calib[r, 7] = self.rho[i] * col_t
            H_calib[r, 8:8 + n_intr] = J_intr

        Hs = [H_pk, H_pk1, H_board, H_calib]
        if H_h is not None:
            Hs.append(H_h)
        return err, Hs

    def _numerical_jacobians(self, pose_k, pose_k1, T_BinW, xi, h=None,
                             eps: float = 1e-6):
        if not self.uses_heights_key:
            _, blocks = numerical_jacobian(
                lambda a, b, c, d: self.error(a, b, c, d),
                [pose_k, pose_k1, T_BinW, xi], eps)
            return blocks
        h = np.zeros(self.n_bp) if h is None else np.asarray(h, float)
        _, blocks = numerical_jacobian(
            lambda a, b, c, d, e: self.error(a, b, c, d, e),
            [pose_k, pose_k1, T_BinW, xi, h], eps)
        return blocks

    # ------------------------------------------------------------------
    def make_factor(self, k_pose0, k_pose1, k_board, k_calib, k_heights=None,
                    sigma_px: float = 1.0, huber_px: float = 0.0):
        """4-key gtsam.CustomFactor (5 with the board-heights key; isotropic
        pixel noise, optional Huber)."""
        self.keys = [k_pose0, k_pose1, k_board, k_calib]
        if self.uses_heights_key:
            assert k_heights is not None, "board_pids needs the board-heights key"
            self.keys.append(k_heights)
        keys = gtsam.KeyVector(self.keys)
        noise = gtsam.noiseModel.Isotropic.Sigma(2 * self.n, sigma_px)
        if huber_px > 0:
            noise = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(huber_px / sigma_px),
                noise)

        def err_func(this, values, H):
            pose_k = values.atPose3(k_pose0)
            pose_k1 = values.atPose3(k_pose1)
            board = values.atPose3(k_board)
            xi = values.atVector(k_calib)
            h = values.atVector(k_heights) if self.uses_heights_key else None
            if H is None:
                return self.error(pose_k, pose_k1, board, xi, h)
            e, Js = self.error_and_jacobians(pose_k, pose_k1, board, xi, h)
            for i in range(len(Js)):
                H[i] = Js[i]
            return e

        return gtsam.CustomFactor(noise, keys, err_func)
