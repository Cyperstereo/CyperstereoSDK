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
"""Interpolated-pose reprojection over WORLD-FRAME LANDMARK VARIABLES.

Feature-mode counterpart of `CameraCalibInterpFactor`: identical interpolated
camera geometry and `CameraCalib` (ξ16) handling, but the 3-D point is an
ESTIMATED `Point3` landmark in the world frame instead of a known board corner
transformed by a board pose. One factor per frame, stacked over that frame's
corners; keys are [pose_k, pose_k1, calib(ξ16), L_0, …, L_{N-1}] with one
landmark key per corner.

This is the only structural difference between board-based and feature-based
joint calibration: the board factor reprojects
`T_BinW · p_board[id]` (known point + one shared 6-DOF board pose); this factor
reprojects `L[id]` directly (a 3-DOF landmark per corner). The pose_k / pose_k1
/ calib Jacobians are the SAME analytic blocks; the board block (2N×6) becomes
per-landmark blocks (2×3): d(reproj)/d(L) = J_point (Point3 retract is additive).
"""

from __future__ import annotations

import numpy as np
import gtsam

from .camera_calib import CameraCalib
from .projection import make_projector
from .pose_interpolation import interp_pose, interp_pose_tangents
from .corner_reproj import CornerReprojMixin
from .lie_groups import se3_right_jacobian
from .numdiff import numerical_jacobian

# default depth margin (m): landmarks within this depth of the camera principal
# plane are treated as grazing (skipped) — the cheirality check covers z<0, this
# covers the 0<z<z_min grazing band where pinhole x/z explodes (1e8 px) and
# derails LM. This is a FALLBACK only; in practice the pipeline sets a per-camera
# adaptive margin (z_min = frac * median seed depth) via the `z_min` ctor arg, so
# the threshold scales with scene depth instead of being hard-coded.
_Z_MIN = 0.5


class CameraFeatureInterpFactor(CornerReprojMixin):
    """Per-frame stacked reprojection over [pose_k, pose_k1, calib(ξ16), L_i…]."""

    def __init__(self, uv_meas, point_ids, dt_frame: float, dt_interval: float,
                 calib_base: CameraCalib, image_size, rolling_mode: int = 0,
                 z_min: float = _Z_MIN):
        self.uv = np.asarray(uv_meas, float).reshape(-1, 2)
        self.ids = [int(i) for i in point_ids]
        assert len(self.uv) == len(self.ids), "one landmark id per measurement"
        assert dt_interval > 0
        self.dt_frame = float(dt_frame)
        self.dt = float(dt_interval)
        self.base = calib_base
        self.w_px, self.h_px = int(image_size[0]), int(image_size[1])
        self.rolling_mode = int(rolling_mode)
        self.n = len(self.uv)
        self.z_min = float(z_min)        # depth-margin guard (m); see module note
        self.rho = self._row_offsets()

    def seed_depths(self, pose_k, pose_k1, P_world):
        """Per-corner landmark depth (m) in the camera frame at the BASE calib
        (xi=0). Used by the pipeline to set an adaptive per-camera `z_min` from
        the seed scene scale before the solve. Returns (n,) array."""
        calib = self.base
        P_world = np.asarray(P_world, float).reshape(-1, 3)
        out = np.empty(self.n)
        for i in range(self.n):
            T_I, _phi, _A, _R = interp_pose(pose_k, pose_k1, self._lam(calib, i))
            out[i] = T_I.compose(calib.T).transformTo(gtsam.Point3(*P_world[i]))[2]
        return out

    # ------------------------------------------------------------------
    def error(self, pose_k, pose_k1, xi, P_world):
        """Stacked 2N reprojection residual (unwhitened). P_world: (N,3)."""
        return self._residuals(pose_k, pose_k1, xi, P_world)[0]

    def _residuals(self, pose_k, pose_k1, xi, P_world):
        """(stacked 2N unwhitened residual, per-corner validity mask). Invalid
        corners (depth-margin / cheirality guard) keep zero rows in the
        residual but are False in the mask so stats can exclude them."""
        calib = self.base.retract(np.asarray(xi, float))
        proj = make_projector(calib.model, calib.intr)
        P_world = np.asarray(P_world, float).reshape(-1, 3)
        err = np.zeros(2 * self.n)
        valid = np.zeros(self.n, bool)
        T_I_cache = None
        for i in range(self.n):
            if calib.tr == 0.0 and T_I_cache is not None:
                T_I = T_I_cache
            else:
                T_I, _phi, _A, _R = interp_pose(pose_k, pose_k1,
                                                self._lam(calib, i))
                if calib.tr == 0.0:
                    T_I_cache = T_I
            T_CtoW = T_I.compose(calib.T)
            if T_CtoW.transformTo(gtsam.Point3(*P_world[i]))[2] < self.z_min:
                continue                         # grazing/behind: depth-margin guard
            try:
                px, _, _, _ = proj.project(T_CtoW, P_world[i], jac=False)
            except RuntimeError:
                continue
            err[2 * i:2 * i + 2] = px - self.uv[i]
            valid[i] = True
        return err, valid

    def camera_points_values(self, values: gtsam.Values) -> np.ndarray:
        """Each landmark in the camera frame at a gtsam Values (N x 3), by the same chain as
        `_residuals`; for the corner accounting, which needs the angle."""
        pose_k, pose_k1, xi, P_world = self.args_from_values(values)
        calib = self.base.retract(np.asarray(xi, float))
        out = np.empty((self.n, 3))
        for i in range(self.n):
            T_I = interp_pose(pose_k, pose_k1, self._lam(calib, i))[0]
            out[i] = np.asarray(T_I.compose(calib.T).transformTo(gtsam.Point3(*P_world[i])), float)
        return out

    def args_from_values(self, values: gtsam.Values):
        """(pose_k, pose_k1, xi, P_world) at a gtsam Values, using the keys from make_factor."""
        k = self.keys
        return (values.atPose3(k[0]), values.atPose3(k[1]), values.atVector(k[2]),
                np.array([values.atPoint3(q) for q in k[3:]]))

    # ------------------------------------------------------------------
    def error_and_jacobians(self, pose_k, pose_k1, xi, P_world):
        """(error 2N, [H_pose_k, H_pose_k1, H_calib, H_L_0, …, H_L_{N-1}])."""
        xi = np.asarray(xi, float)
        calib = self.base.retract(xi)
        proj = make_projector(calib.model, calib.intr)
        n_intr = calib.n_intr
        T_C = calib.T
        Ad_TCinv = T_C.inverse().AdjointMap()
        Jr_xiT = se3_right_jacobian(xi[0:6])
        P_world = np.asarray(P_world, float).reshape(-1, 3)

        p_k = np.asarray(pose_k.translation(), float)
        p_k1 = np.asarray(pose_k1.translation(), float)
        dp = p_k1 - p_k

        err = np.zeros(2 * self.n)
        H_pk = np.zeros((2 * self.n, 6))
        H_pk1 = np.zeros((2 * self.n, 6))
        H_calib = np.zeros((2 * self.n, calib.DIM))
        H_L = [np.zeros((2 * self.n, 3)) for _ in range(self.n)]

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
            if T_CtoW.transformTo(gtsam.Point3(*P_world[i]))[2] < self.z_min:
                continue                         # grazing/behind: depth-margin guard
            try:
                px, J_pose, J_point, J_intr = proj.project(T_CtoW, P_world[i])
            except RuntimeError:
                continue
            err[2 * i:2 * i + 2] = px - self.uv[i]
            r = slice(2 * i, 2 * i + 2)
            J_T = J_pose @ Ad_TCinv

            H_pk[r] = J_T @ M_k
            H_pk1[r] = J_T @ M_k1
            H_calib[r, 0:6] = J_pose @ Jr_xiT
            col_t = J_T @ zeta_lam / self.dt
            H_calib[r, 6] = col_t
            H_calib[r, 7] = self.rho[i] * col_t
            H_calib[r, 8:8 + n_intr] = J_intr
            # world-frame landmark: p_W = L, Point3 retract additive -> d/dL = I
            H_L[i][r, :] = J_point

        return err, [H_pk, H_pk1, H_calib, *H_L]

    def _numerical_jacobians(self, pose_k, pose_k1, xi, P_world, eps=1e-6):
        _, blocks = numerical_jacobian(
            lambda a, b, c, d: self.error(a, b, c, d),
            [pose_k, pose_k1, xi, np.asarray(P_world, float)], eps)
        return blocks

    # ------------------------------------------------------------------
    def make_factor(self, k_pose0, k_pose1, k_calib, landmark_keys,
                    sigma_px: float = 1.0, huber_px: float = 0.0):
        """gtsam.CustomFactor over [pose0, pose1, calib, L_0…L_{N-1}]."""
        assert len(landmark_keys) == self.n
        keys = gtsam.KeyVector([k_pose0, k_pose1, k_calib, *landmark_keys])
        self.keys = [k_pose0, k_pose1, k_calib, *landmark_keys]
        noise = gtsam.noiseModel.Isotropic.Sigma(2 * self.n, sigma_px)
        if huber_px > 0:
            noise = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(huber_px / sigma_px),
                noise)

        def err_func(this, values, H):
            pose_k = values.atPose3(k_pose0)
            pose_k1 = values.atPose3(k_pose1)
            xi = values.atVector(k_calib)
            P = np.array([values.atPoint3(lk) for lk in landmark_keys])
            if H is None:
                return self.error(pose_k, pose_k1, xi, P)
            e, Js = self.error_and_jacobians(pose_k, pose_k1, xi, P)
            for i in range(len(Js)):
                H[i] = Js[i]
            return e

        return gtsam.CustomFactor(noise, keys, err_func)
