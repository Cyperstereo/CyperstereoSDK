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
"""ImuFactorACI_In_SE3 — single-IMU ACI factor with intrinsics.

Pure-Python port of the C++ MVIS `graph_IMU/ImuFactorACI_In_SE3` (a
NoiseModelFactor5<Pose3,Pose3,Vector9,Vector9,ImuIntrinsics>). The custom 24-dim
`ImuIntrinsics` manifold can't live in Python `Values`, so the intrinsics
variable is stored as its 24-dim minimal-tangent `Vector` ξ around a fixed base
(the preintegration linearization point `intr_init`); the factor evaluates
`intr_init.retract(ξ)`. This keeps it a 5-key `gtsam.CustomFactor` — one cohesive
intrinsics variable, matching the C++ signature:

    keys = [pose0, pose1, vb0, vb1, intr]
      pose0,pose1 : Pose3   IMU pose in global (I_k -> G)
      vb0,vb1     : Vector9 [v_inG(3), bias_gyro(3), bias_accel(3)]
      intr        : Vector24 ξ — tangent of ImuIntrinsics at intr_init

Because `base = intr_init`, the manifold axiom gives the intrinsic deviation
Δ_in = local(intr_init, intr_init.retract(ξ)) = ξ exactly, so the residual's
intrinsics enter purely through Δ_in = ξ and the intrinsic Jacobian is just
`par_meas_in` (no retract chain).

Residual (15-dim, order [θ, p, v, b_g, b_a]) is the bias/intrinsic-corrected ACI
prediction error, whitened by aci.whitener_rot_corrected (Unit noise; whitening is internal).
"""

from __future__ import annotations

import numpy as np
import gtsam

from mvis.calibration.factors.lie_groups import (
    skew, so3_left_jacobian, exp_so3, log_so3,
)
from mvis.imu import ImuModel
from mvis.calibration.factors.aci import whitener_rot_corrected

_I3 = np.eye(3)


class ImuFactorACI_In:
    """Single-IMU ACI factor data + residual/Jacobian evaluation (single intr key)."""

    def __init__(self, D_R, D_v, D_p, P_meas, Dt, g_inG,
                 bg_init, ba_init, intr_init: ImuModel,
                 J_R_bg, J_R_ba, J_p_bg, J_p_ba, J_v_bg, J_v_ba,
                 J_R_in, J_p_in, J_v_in,
                 J_R_tga=None, J_p_tga=None, J_v_tga=None):
        self.D_R = np.asarray(D_R, float)
        self.D_v = np.asarray(D_v, float)
        self.D_p = np.asarray(D_p, float)
        self.P_meas = np.asarray(P_meas, float)
        self.Dt = float(Dt)
        self.g_inG = np.asarray(g_inG, float)
        self.bg_init = np.asarray(bg_init, float)
        self.ba_init = np.asarray(ba_init, float)
        self.intr_init = intr_init           # base for the intrinsics tangent ξ
        self.model = intr_init.model
        self.J_R_bg, self.J_R_ba = np.asarray(J_R_bg), np.asarray(J_R_ba)
        self.J_p_bg, self.J_p_ba = np.asarray(J_p_bg), np.asarray(J_p_ba)
        self.J_v_bg, self.J_v_ba = np.asarray(J_v_bg), np.asarray(J_v_ba)
        self.J_R_in = np.asarray(J_R_in)
        self.J_p_in = np.asarray(J_p_in)
        self.J_v_in = np.asarray(J_v_in)
        # gyro<->accel time offset (optional 6th key; zeros = t_ga insensitive)
        self.J_R_tga = np.zeros(3) if J_R_tga is None else np.asarray(J_R_tga, float)
        self.J_p_tga = np.zeros(3) if J_p_tga is None else np.asarray(J_p_tga, float)
        self.J_v_tga = np.zeros(3) if J_v_tga is None else np.asarray(J_v_tga, float)
        # t_ga the integration was done at; the J_*_tga terms cover t_ga - t_lin (tga_relin.py)
        self.t_lin = 0.0

    @classmethod
    def from_preintegrator(cls, pre, g_inG):
        """Build the factor from a finished preintegrator (its D_R/D_v/D_p, measurement covariance, Dt, init biases) + gravity g_inG."""
        return cls(
            D_R=pre.D_R, D_v=pre.D_v, D_p=pre.D_p, P_meas=pre.P_meas,
            Dt=pre.Dt, g_inG=g_inG, bg_init=pre.b_g_init, ba_init=pre.b_a_init,
            intr_init=pre.intr,
            J_R_bg=pre.J_R_bg, J_R_ba=pre.J_R_ba,
            J_p_bg=pre.J_p_bg, J_p_ba=pre.J_p_ba,
            J_v_bg=pre.J_v_bg, J_v_ba=pre.J_v_ba,
            J_R_in=pre.J_R_in, J_p_in=pre.J_p_in, J_v_in=pre.J_v_in,
            J_R_tga=getattr(pre, 'J_R_tga', None),
            J_p_tga=getattr(pre, 'J_p_tga', None),
            J_v_tga=getattr(pre, 'J_v_tga', None))

    # ------------------------------------------------------------------
    def _common(self, pose0, pose1, vb0, vb1, xi_in, t_ga=0.0):
        """Residual + reusable intermediates. xi_in = ImuIntrinsics tangent (=Δ_in);
        t_ga = gyro<->accel time offset (accel at gyro time t is a_m(t + t_ga))."""
        R_I0toG = pose0.rotation().matrix()
        R_GtoI0 = R_I0toG.T
        p0 = pose0.translation()
        R_I1toG = pose1.rotation().matrix()
        p1 = pose1.translation()
        vb0 = np.asarray(vb0, float); vb1 = np.asarray(vb1, float)
        v0, v1 = vb0[0:3], vb1[0:3]
        bias0, bias1 = vb0[3:9], vb1[3:9]
        bg0, ba0 = bias0[0:3], bias0[3:6]
        g, Dt = self.g_inG, self.Dt

        D_R_hat = R_GtoI0 @ R_I1toG
        D_v_hat = R_GtoI0 @ (v1 - v0 - g * Dt)
        D_p_hat = R_GtoI0 @ (p1 - p0 - v0 * Dt - 0.5 * g * Dt ** 2)
        b_hat = bias1 - bias0

        d_bg = bg0 - self.bg_init
        d_ba = ba0 - self.ba_init
        d_in = np.asarray(xi_in, float)      # Δ_in = ξ exactly (base = intr_init)

        t_ga = float(t_ga) - self.t_lin
        d_th_total = (self.J_R_bg @ d_bg + self.J_R_ba @ d_ba
                      + self.J_R_in @ d_in + self.J_R_tga * t_ga)
        R_bcorr = exp_so3(d_th_total)
        D_R_hat_c = D_R_hat @ R_bcorr.T
        D_v_hat_c = (D_v_hat - self.J_v_bg @ d_bg - self.J_v_ba @ d_ba
                     - self.J_v_in @ d_in - self.J_v_tga * t_ga)
        D_p_hat_c = (D_p_hat - self.J_p_bg @ d_bg - self.J_p_ba @ d_ba
                     - self.J_p_in @ d_in - self.J_p_tga * t_ga)

        err_th = log_so3(self.D_R.T @ D_R_hat_c)
        err_p = D_p_hat_c - self.D_p
        err_v = D_v_hat_c - self.D_v
        err = np.concatenate([err_th, err_p, err_v, b_hat])

        ctx = dict(R_GtoI0=R_GtoI0, R_I1toG=R_I1toG, D_R_hat_c=D_R_hat_c,
                   D_p_hat=D_p_hat, D_v_hat=D_v_hat, R_bcorr=R_bcorr,
                   d_th_total=d_th_total, Dt=Dt)
        return err, ctx

    def whitened_error(self, pose0, pose1, vb0, vb1, xi_in, t_ga=0.0):
        """Whitened 15-dim IMU preintegration residual: Sigma^-1/2 (predicted - measured) between [pose0,vb0] and [pose1,vb1] given the IMU-intrinsic tangent xi_in."""
        err, ctx = self._common(pose0, pose1, vb0, vb1, xi_in, t_ga)
        return whitener_rot_corrected(self.P_meas, ctx["d_th_total"]) @ err

    def error_and_jacobians(self, pose0, pose1, vb0, vb1, xi_in, t_ga=None):
        """Returns (whitened error, [H_p0, H_p1, H_vb0, H_vb1, H_intr]); with t_ga
        (scalar) given, appends the 15x1 H_tga block."""
        err, c = self._common(pose0, pose1, vb0, vb1, xi_in,
                              0.0 if t_ga is None else t_ga)
        # err_th differentiated through Exp(d_th_total) -> LEFT Jacobian at err=0
        # (gtsam Hamilton/right-retract; C++ uses Jr under OpenVINS JPL convention).
        # Computed once and shared with the covariance correction.
        Jl_dth = so3_left_jacobian(c["d_th_total"])
        W = whitener_rot_corrected(self.P_meas, c["d_th_total"], Jl_dth)
        R_GtoI0, R_I1toG = c["R_GtoI0"], c["R_I1toG"]

        H_p0 = np.zeros((15, 6))
        H_p0[0:3, 0:3] = -c["D_R_hat_c"].T
        H_p0[3:6, 0:3] = skew(c["D_p_hat"]); H_p0[3:6, 3:6] = -_I3
        H_p0[6:9, 0:3] = skew(c["D_v_hat"])

        H_p1 = np.zeros((15, 6))
        H_p1[0:3, 0:3] = c["R_bcorr"]
        H_p1[3:6, 3:6] = R_GtoI0 @ R_I1toG

        H_vb0 = np.zeros((15, 9))
        H_vb0[0:3, 3:6] = -Jl_dth @ self.J_R_bg
        H_vb0[0:3, 6:9] = -Jl_dth @ self.J_R_ba   # g-sensitivity coupling (C++ omits)
        H_vb0[3:6, 0:3] = -R_GtoI0 * c["Dt"]
        H_vb0[3:6, 3:6] = -self.J_p_bg; H_vb0[3:6, 6:9] = -self.J_p_ba
        H_vb0[6:9, 0:3] = -R_GtoI0
        H_vb0[6:9, 3:6] = -self.J_v_bg; H_vb0[6:9, 6:9] = -self.J_v_ba
        H_vb0[9:12, 3:6] = -_I3
        H_vb0[12:15, 6:9] = -_I3

        H_vb1 = np.zeros((15, 9))
        H_vb1[6:9, 0:3] = R_GtoI0
        H_vb1[9:12, 3:6] = _I3
        H_vb1[12:15, 6:9] = _I3

        # H intrinsics (15x24) w.r.t. ξ_in — Δ_in = ξ_in so no retract chain.
        H_intr = np.zeros((15, 24))
        H_intr[0:3, :] = -Jl_dth @ self.J_R_in
        H_intr[3:6, :] = -self.J_p_in
        H_intr[6:9, :] = -self.J_v_in

        H = [W @ H_p0, W @ H_p1, W @ H_vb0, W @ H_vb1, W @ H_intr]
        if t_ga is not None:
            H_tga = np.zeros((15, 1))
            H_tga[0:3, 0] = -Jl_dth @ self.J_R_tga
            H_tga[3:6, 0] = -self.J_p_tga
            H_tga[6:9, 0] = -self.J_v_tga
            H.append(W @ H_tga)
        return W @ err, H

    # ------------------------------------------------------------------
    def make_factor(self, k_pose0, k_pose1, k_vb0, k_vb1, k_intr, k_tga=None):
        """Build the gtsam.CustomFactor (Unit noise; whitening is internal).

        k_intr holds the 24-dim ImuIntrinsics tangent ξ around intr_init.
        k_tga (optional) holds the scalar gyro<->accel time offset t_ga as a
        Vector1 — omitted, the factor is the byte-identical 5-key version.
        """
        ks = [k_pose0, k_pose1, k_vb0, k_vb1, k_intr]
        if k_tga is not None:
            ks.append(k_tga)
        keys = gtsam.KeyVector(ks)
        noise = gtsam.noiseModel.Unit.Create(15)

        def err_func(this, values, H):
            pose0 = values.atPose3(k_pose0)
            pose1 = values.atPose3(k_pose1)
            vb0 = values.atVector(k_vb0)
            vb1 = values.atVector(k_vb1)
            xi_in = values.atVector(k_intr)
            t_ga = float(values.atVector(k_tga)[0]) if k_tga is not None else None
            if H is None:                      # error-only (LM lambda trials):
                return self.whitened_error(    # same _common + W path, no blocks
                    pose0, pose1, vb0, vb1, xi_in, 0.0 if t_ga is None else t_ga)
            e, Js = self.error_and_jacobians(pose0, pose1, vb0, vb1, xi_in, t_ga)
            for i in range(len(Js)):
                H[i] = Js[i]
            return e

        return gtsam.CustomFactor(noise, keys, err_func)
