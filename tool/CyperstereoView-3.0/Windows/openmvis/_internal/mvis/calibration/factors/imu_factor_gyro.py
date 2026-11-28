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
"""ImuFactorACI_Multi_In_SO3 — gyroscope-only ACI factor (IMU-gyro calibration).

Pure-Python port of the C++ MVIS `graph_IMU/ImuFactorACI_Multi_In_SO3`
(NoiseModelFactor5<Pose3,Pose3,Vector3,Vector3,ImuCalib>). Rotation-only: the
6-dim residual is [err_theta(3), err_bg(3)]. The aux gyro orientation is derived
from the base trajectory via the extrinsic + time-offset coupling; only the gyro
intrinsic `x_Dw` (6) and the extrinsic rotation / td of the ImuCalib affect it.

Decomposed (ImuCalib as a tangent Vector around base_calib) -> 5-key CustomFactor:

    keys = [pose0, pose1, bg0, bg1, calib]
      pose0,pose1 : Pose3    BASE IMU pose in global (I_k -> G)
      bg0,bg1     : Vector3  AUX gyro bias
      calib       : Vector31 ξ — tangent of ImuCalib at base_calib

Jacobians: ANALYTIC (rotation-only chain rule, sharing `imu_aux_pose_jacobians`
with the multi-IMU factor); `_numerical_jacobians` is the central-difference
cross-check used in tests.
"""

from __future__ import annotations

import numpy as np
import gtsam

from mvis.calibration.factors.imu_calib import ImuCalib
from mvis.calibration.factors.lie_groups import (
    so3_left_jacobian, se3_right_jacobian, exp_so3, log_so3,
)
from mvis.calibration.factors.pose_interpolation import (
    imu_aux_rotation, imu_aux_pose_jacobians,
)
from mvis.calibration.factors.numdiff import numerical_jacobian
from mvis.calibration.factors.aci import whitener_rot_corrected


class ImuFactorACI_Multi_In_SO3:
    """Gyro-only ACI factor: base trajectory + extrinsic/td/Dw -> gyro rotation."""

    def __init__(self, D_R, P_meas, Dt, bg_init, intr_init,
                 J_R_bg, J_R_in, w_inIs0, w_inIs1, base_calib: ImuCalib,
                 window_td: float = 0.0):
        self.D_R = np.asarray(D_R, float)
        self.P_meas = np.asarray(P_meas, float)       # 6x6 [theta, bg]
        self.Dt = float(Dt)
        self.bg_init = np.asarray(bg_init, float)
        self.intr_init = intr_init
        self.J_R_bg = np.asarray(J_R_bg, float)       # 3x3
        self.J_R_in = np.asarray(J_R_in, float)       # 3x6 (w.r.t. x_Dw)
        self.w_inIs0 = np.asarray(w_inIs0, float)
        self.w_inIs1 = np.asarray(w_inIs1, float)
        self.base_calib = base_calib
        # window_td: offset already absorbed by shifting the aux window (as in
        # ImuFactorACI_Multi_In_V); the factor propagates only calib.td - window_td.
        self.window_td = float(window_td)

    @classmethod
    def from_preintegrator(cls, pre, T_ext_init, td_init, window_td: float = 0.0):
        """pre = ACIGyroPreintegrator fed the AUX gyro interval (aux frame).
        window_td = offset already baked into the aux window slice."""
        base_calib = ImuCalib(T_ext_init, td_init, pre.intr)
        return cls(pre.D_R, pre.P_meas, pre.Dt, pre.b_g_init, pre.intr,
                   pre.J_R_bg, pre.J_R_in, pre.w_m_s, pre.w_m_e, base_calib,
                   window_td=window_td)

    # ------------------------------------------------------------------
    def _aux_rotations(self, pose0, pose1, calib: ImuCalib):
        R, td = calib.R_IstoI, calib.td - self.window_td   # RESIDUAL
        return (imu_aux_rotation(pose0, R, td, self.w_inIs0),
                imu_aux_rotation(pose1, R, td, self.w_inIs1))

    def _common(self, pose0, pose1, bg0, bg1, xi_calib):
        calib = self.base_calib.retract(xi_calib)
        bg0 = np.asarray(bg0, float); bg1 = np.asarray(bg1, float)
        R_Is0toG, R_Is1toG = self._aux_rotations(pose0, pose1, calib)

        D_R_hat = R_Is0toG.T @ R_Is1toG
        d_bg = bg0 - self.bg_init
        d_in = calib.imu_in.x_dw - self.intr_init.x_dw          # gyro Dw deviation (6)
        d_th = self.J_R_bg @ d_bg + self.J_R_in @ d_in
        R_bcorr = exp_so3(d_th)
        D_R_hat_c = D_R_hat @ R_bcorr.T

        err_th = log_so3(self.D_R.T @ D_R_hat_c)
        err_b = bg1 - bg0
        err = np.concatenate([err_th, err_b])
        ctx = dict(calib=calib, d_th=d_th, R_bcorr=R_bcorr, D_R_hat_c=D_R_hat_c)
        return err, ctx

    def whitened_error(self, pose0, pose1, bg0, bg1, xi_calib):
        """Whitened 6-dim gyro-only preintegration residual (rotation + gyro bias) — uses no accelerometer."""
        err, c = self._common(pose0, pose1, bg0, bg1, xi_calib)
        return whitener_rot_corrected(self.P_meas, c["d_th"]) @ err

    def error_and_jacobians(self, pose0, pose1, bg0, bg1, xi_calib):
        """Whitened error + [H_p0(6x6), H_p1(6x6), H_bg0(6x3), H_bg1(6x3),
        H_calib(6x31)] — ANALYTIC (rotation-only chain rule).

        Residual rows [err_theta(3), err_bg(3)]. err_theta depends on the aux
        rotations (chained to base-pose rotation + extrinsic/td) and on bg/Dw via
        the bias/intrinsic correction; err_bg = bg1 - bg0. Uses the left Jacobian
        -Jl(d_theta) (gtsam convention). `_numerical_jacobians` is the cross-check.
        """
        bg0 = np.asarray(bg0, float); bg1 = np.asarray(bg1, float)
        xi_calib = np.asarray(xi_calib, float)
        # ONE residual pass (the earlier code evaluated `_common` twice —
        # once through whitened_error — and the covariance correction twice);
        # the whitened error is the same W @ err whitened_error returns.
        err, c = self._common(pose0, pose1, bg0, bg1, xi_calib)
        calib, d_th = c["calib"], c["d_th"]
        R_bcorr, D_R_hat_c = c["R_bcorr"], c["D_R_hat_c"]
        R_IstoI, td = calib.R_IstoI, calib.td - self.window_td   # RESIDUAL
        Jl = so3_left_jacobian(d_th)
        W = whitener_rot_corrected(self.P_meas, d_th, Jl)
        e0 = W @ err
        I3 = np.eye(3)

        # residual-w.r.t-aux-rotation blocks (rotation-only analogue of single-IMU)
        par_meas_rots0 = -D_R_hat_c.T
        par_meas_rots1 = R_bcorr

        def rot_coupling(pose, w_inIs):
            # rotation rows of the shared aux-pose coupling (gyro is rotation-only;
            # p_IsinI / v_base don't affect rows 0:3, so pass dummies)
            par_p, _, par_calib = imu_aux_pose_jacobians(
                pose, R_IstoI, calib.p_IsinI, td, w_inIs, np.zeros(3))
            return par_p[0:3, 0:3], par_calib[0:3, :]

        pr0, pc0 = rot_coupling(pose0, self.w_inIs0)
        pr1, pc1 = rot_coupling(pose1, self.w_inIs1)

        H_p0 = np.zeros((6, 6)); H_p0[0:3, 0:3] = par_meas_rots0 @ pr0
        H_p1 = np.zeros((6, 6)); H_p1[0:3, 0:3] = par_meas_rots1 @ pr1

        H_bg0 = np.zeros((6, 3)); H_bg0[0:3, :] = -Jl @ self.J_R_bg; H_bg0[3:6, :] = -I3
        H_bg1 = np.zeros((6, 3)); H_bg1[3:6, :] = I3

        H_calib = np.zeros((6, 31))
        ext_td = par_meas_rots0 @ pc0 + par_meas_rots1 @ pc1     # 3x7 (err_theta rows)
        H_calib[0:3, 0:6] = ext_td[:, 0:6] @ se3_right_jacobian(xi_calib[0:6])
        H_calib[0:3, 6] = ext_td[:, 6]
        n_dw = self.intr_init.n_dw                              # 6 (tri) or 9 (Calyx full)
        H_calib[0:3, 7:7 + n_dw] = -Jl @ self.J_R_in            # gyro Dw cols

        return e0, [W @ H_p0, W @ H_p1, W @ H_bg0, W @ H_bg1, W @ H_calib]

    def _numerical_jacobians(self, pose0, pose1, bg0, bg1, xi_calib):
        """Central-difference Jacobian — cross-check for the analytic."""
        return numerical_jacobian(
            self.whitened_error, [pose0, pose1, bg0, bg1, xi_calib])

    def make_factor(self, k_pose0, k_pose1, k_bg0, k_bg1, k_calib):
        """Build a gtsam.CustomFactor over (pose0,pose1,bg0,bg1,calib) wrapping the gyro-only residual + its analytic Jacobians."""
        keys = gtsam.KeyVector([k_pose0, k_pose1, k_bg0, k_bg1, k_calib])
        noise = gtsam.noiseModel.Unit.Create(6)

        def err_func(this, values, H):
            pose0 = values.atPose3(k_pose0); pose1 = values.atPose3(k_pose1)
            bg0 = values.atVector(k_bg0); bg1 = values.atVector(k_bg1)
            xi_calib = values.atVector(k_calib)
            if H is None:                      # error-only (LM lambda trials):
                return self.whitened_error(pose0, pose1, bg0, bg1, xi_calib)
            e, Js = self.error_and_jacobians(pose0, pose1, bg0, bg1, xi_calib)
            for i in range(5):
                H[i] = Js[i]
            return e

        return gtsam.CustomFactor(noise, keys, err_func)
