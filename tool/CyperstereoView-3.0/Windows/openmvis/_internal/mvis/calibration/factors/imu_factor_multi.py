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
"""ImuFactorACI_Multi_In_V_SE3 — auxiliary-IMU ACI factor (IMU-IMU calibration).

Pure-Python port of the C++ MVIS `graph_IMU/ImuFactorACI_Multi_In_v_SE3`. The C++ factor
is a NoiseModelFactor7<Pose3,Pose3,Vector9,Vector9,Vector9,Vector9,ImuCalib>.
The 31-dim `ImuCalib` (extrinsic + td + intrinsics) can't live in Python Values,
so it is stored as its 31-dim minimal-tangent `Vector` ξ around a fixed base
estimate `base_calib` — making this a 7-key `gtsam.CustomFactor` (matching the
C++ Factor7), with `ImuCalib` kept as one cohesive variable:

    keys = [pose0, pose1, vb0, vb1, vbs0, vbs1, calib]
      pose0,pose1 : Pose3    BASE IMU pose in global (I_k -> G)
      vb0,vb1     : Vector9  base IMU [v_inG(3), b_g(3), b_a(3)]   (only v used)
      vbs0,vbs1   : Vector9  AUX  IMU [b_g(3), b_a(3), v_inG(3)]   (note layout!)
      calib       : Vector31 ξ — tangent of ImuCalib at base_calib
                    [d_pose(6), d_td(1), d_imu_in(24)]

base_calib.imu_in == intr_init (the aux preintegration linearization), so the
intrinsics tangent passed to the inner single-IMU factor is just ξ[7:31].

Insight: the residual is the SAME ACI residual as the single-IMU factor, but on
the AUX IMU's poses/velocities — which are derived from the base trajectory via
a kinematic coupling (extrinsic + time-offset propagation). So we reuse
`ImuFactorACI_In` (built from the aux preintegrator) for the residual, and only
the kinematic coupling + its chained Jacobians are new here.

Kinematic coupling (per endpoint k):
    w_inI   = R_IstoI · w_inIs                       (aux ang.vel -> base frame)
    R_IktoG = R_pose · Exp(w_inI · td)               (base pose propagated by td)
    R_IsktoG= R_IktoG · R_IstoI                       (aux orientation)
    p_IkinG = p_pose + v_base · td
    p_IsinG = p_IkinG + R_IktoG · p_IsinI
    v_IsinG = vbs[6:9]                                (aux velocity, direct state)
"""

from __future__ import annotations

import numpy as np
import gtsam

from mvis.calibration.factors.imu_factor import ImuFactorACI_In
from mvis.calibration.factors.imu_calib import ImuCalib
from mvis.calibration.factors.lie_groups import se3_right_jacobian
from mvis.calibration.factors.pose_interpolation import (
    imu_aux_pose, imu_aux_pose_jacobians,
)
from mvis.calibration.factors.numdiff import numerical_jacobian


class ImuFactorACI_Multi_In_V:
    """Aux-IMU ACI factor: base trajectory + extrinsic/td/intrinsics -> aux ACI."""

    def __init__(self, base_factor: ImuFactorACI_In, w_inIs0, w_inIs1,
                 base_calib: ImuCalib, window_td: float = 0.0):
        # base_factor is an ImuFactorACI_In built from the AUX IMU preintegrator;
        # base_calib is the linearization point for the ImuCalib tangent ξ.
        self.base = base_factor
        self.w_inIs0 = np.asarray(w_inIs0, float)  # corrected aux ang.vel @ start
        self.w_inIs1 = np.asarray(w_inIs1, float)  # corrected aux ang.vel @ end
        self.base_calib = base_calib
        # window_td: offset already absorbed by SHIFTING the aux preintegration
        # window (the caller sliced the aux at base stamps MINUS window_td, so
        # the aux interval physically aligns with the base poses at
        # td = window_td). The factor then propagates the base pose by only the
        # RESIDUAL (calib.td - window_td), keeping the first-order Exp(w*dt)
        # propagation small even when the absolute aux-base offset is large.
        # Jacobian unchanged: d(td - window_td)/d(d_td) = 1.
        self.window_td = float(window_td)

    @classmethod
    def from_preintegrator(cls, pre, g_inG, T_ext_init, td_init,
                           window_td: float = 0.0):
        """pre = ACIPreintegrator fed the AUX IMU interval (in the aux frame).
        T_ext_init/td_init = current aux-base extrinsic / time-offset estimates.
        window_td = offset already baked into the aux window slice (see ctor)."""
        base = ImuFactorACI_In.from_preintegrator(pre, g_inG)
        base_calib = ImuCalib(T_ext_init, td_init, pre.intr)
        return cls(base, pre.w_m_s, pre.w_m_e, base_calib, window_td=window_td)

    # ------------------------------------------------------------------
    def _aux_states(self, pose0, pose1, vb0, vb1, vbs0, vbs1, calib: ImuCalib):
        """Derive aux-IMU (pose, vb) at both endpoints from base + calibration.

        Pose: shared `imu_aux_pose` (time-offset propagation + extrinsic). vb:
        aux layout [b_g, b_a, v] reordered to the single-IMU layout [v, b_g, b_a].
        """
        R, p = calib.R_IstoI, calib.p_IsinI
        td = calib.td - self.window_td       # RESIDUAL (window absorbs the rest)
        vb0 = np.asarray(vb0, float); vb1 = np.asarray(vb1, float)
        vbs0 = np.asarray(vbs0, float); vbs1 = np.asarray(vbs1, float)
        pose_Is0 = imu_aux_pose(pose0, R, p, td, self.w_inIs0, vb0[0:3])
        pose_Is1 = imu_aux_pose(pose1, R, p, td, self.w_inIs1, vb1[0:3])
        vb_Is0 = np.concatenate([vbs0[6:9], vbs0[0:3], vbs0[3:6]])  # [v, bg, ba]
        vb_Is1 = np.concatenate([vbs1[6:9], vbs1[0:3], vbs1[3:6]])
        return pose_Is0, pose_Is1, vb_Is0, vb_Is1

    def whitened_error(self, pose0, pose1, vb0, vb1, vbs0, vbs1, xi_calib,
                       t_ga=0.0):
        """Whitened preintegration residual for a non-base (aux) IMU, including its lever-arm + time-offset coupling to the base IMU.

        t_ga = the AUX IMU's own internal gyro<->accel offset (its accel at aux
        gyro time t is a_m(t + t_ga)); a pure preintegration term, independent
        of the base->aux kinematic coupling."""
        calib = self.base_calib.retract(xi_calib)
        pose_Is0, pose_Is1, vb_Is0, vb_Is1 = self._aux_states(
            pose0, pose1, vb0, vb1, vbs0, vbs1, calib)
        # base_calib.imu_in == intr_init, so the intrinsics tangent is ξ[7:31].
        xi_in = np.asarray(xi_calib, float)[7:31]
        return self.base.whitened_error(pose_Is0, pose_Is1, vb_Is0, vb_Is1,
                                        xi_in, t_ga)

    def error_and_jacobians(self, pose0, pose1, vb0, vb1, vbs0, vbs1, xi_calib,
                            t_ga=None):
        """Whitened error + 7 Jacobian blocks (8 with t_ga) — ANALYTIC.

        With t_ga (scalar) given, appends the aux IMU's own 15x1 H_tga column —
        the inner single-IMU factor's preintegration column, which needs no
        base->aux chain-rule (it does not touch the pose coupling).

        Chain rule: the single-IMU factor's analytic Jacobian w.r.t. the aux
        states (`base.error_and_jacobians`) composed with the kinematic-coupling
        Jacobian (`imu_aux_pose_jacobians`, shared with the gyro factor). The calib extrinsic
        columns carry the SE(3) retract chain `Jr_se3(ξ[0:6])` since the calib
        variable is a tangent ξ around base_calib. First-order (exact at
        residual≈0), like the single-IMU factor it reuses. `_numerical_jacobians`
        is the central-difference cross-check used in tests.
        """
        vb0 = np.asarray(vb0, float); vb1 = np.asarray(vb1, float)
        vbs0 = np.asarray(vbs0, float); vbs1 = np.asarray(vbs1, float)
        xi_calib = np.asarray(xi_calib, float)

        calib = self.base_calib.retract(xi_calib)
        pose_Is0, pose_Is1, vb_Is0, vb_Is1 = self._aux_states(
            pose0, pose1, vb0, vb1, vbs0, vbs1, calib)
        xi_in = xi_calib[7:31]
        e0, Hb = self.base.error_and_jacobians(pose_Is0, pose_Is1, vb_Is0, vb_Is1,
                                               xi_in, t_ga)
        H_pIs0, H_pIs1, H_vbIs0, H_vbIs1, H_intr = Hb[:5]

        R, p_ = calib.R_IstoI, calib.p_IsinI
        td = calib.td - self.window_td       # RESIDUAL (window absorbs the rest)
        pp0, pv0, pc0 = imu_aux_pose_jacobians(pose0, R, p_, td, self.w_inIs0, vb0[0:3])
        pp1, pv1, pc1 = imu_aux_pose_jacobians(pose1, R, p_, td, self.w_inIs1, vb1[0:3])

        H_p0 = H_pIs0 @ pp0
        H_p1 = H_pIs1 @ pp1
        H_vb0 = np.zeros((15, 9)); H_vb0[:, 0:3] = H_pIs0 @ pv0   # base bias cols stay 0
        H_vb1 = np.zeros((15, 9)); H_vb1[:, 0:3] = H_pIs1 @ pv1
        # aux vbs layout [bg,ba,v] = single vb_Is [v,bg,ba] reordered
        H_vbs0 = np.column_stack([H_vbIs0[:, 3:6], H_vbIs0[:, 6:9], H_vbIs0[:, 0:3]])
        H_vbs1 = np.column_stack([H_vbIs1[:, 3:6], H_vbIs1[:, 6:9], H_vbIs1[:, 0:3]])

        H_calib = np.zeros((15, 31))
        ext_td = H_pIs0 @ pc0 + H_pIs1 @ pc1                      # 15x7
        H_calib[:, 0:6] = ext_td[:, 0:6] @ se3_right_jacobian(xi_calib[0:6])
        H_calib[:, 6] = ext_td[:, 6]
        H_calib[:, 7:31] = H_intr
        Js = [H_p0, H_p1, H_vb0, H_vb1, H_vbs0, H_vbs1, H_calib]
        if t_ga is not None:
            Js.append(Hb[5])                  # H_tga: pure preintegration column
        return e0, Js

    def _numerical_jacobians(self, pose0, pose1, vb0, vb1, vbs0, vbs1, xi_calib):
        """Central-difference Jacobian — reference for cross-checking the analytic."""
        return numerical_jacobian(
            self.whitened_error, [pose0, pose1, vb0, vb1, vbs0, vbs1, xi_calib])

    def make_factor(self, k_pose0, k_pose1, k_vb0, k_vb1, k_vbs0, k_vbs1, k_calib,
                    k_tga=None):
        """Build the 7-key gtsam.CustomFactor (8 keys with k_tga). k_calib holds
        the 31-dim ImuCalib tangent ξ around base_calib; vbs0/vbs1 are aux
        [b_g, b_a, v] Vector9s. k_tga (optional) holds the AUX IMU's own scalar
        gyro<->accel offset as a Vector1 — omitted, the factor is the
        byte-identical 7-key version."""
        ks = [k_pose0, k_pose1, k_vb0, k_vb1, k_vbs0, k_vbs1, k_calib]
        if k_tga is not None:
            ks.append(k_tga)
        keys = gtsam.KeyVector(ks)
        noise = gtsam.noiseModel.Unit.Create(15)

        def err_func(this, values, H):
            pose0 = values.atPose3(k_pose0); pose1 = values.atPose3(k_pose1)
            vb0 = values.atVector(k_vb0); vb1 = values.atVector(k_vb1)
            vbs0 = values.atVector(k_vbs0); vbs1 = values.atVector(k_vbs1)
            xi_calib = values.atVector(k_calib)
            t_ga = float(values.atVector(k_tga)[0]) if k_tga is not None else None
            if H is None:                      # error-only (LM lambda trials):
                return self.whitened_error(    # skip the block chain
                    pose0, pose1, vb0, vb1, vbs0, vbs1, xi_calib,
                    0.0 if t_ga is None else t_ga)
            e, Js = self.error_and_jacobians(pose0, pose1, vb0, vb1, vbs0, vbs1,
                                             xi_calib, t_ga)
            for i in range(len(Js)):
                H[i] = Js[i]
            return e

        return gtsam.CustomFactor(noise, keys, err_func)
