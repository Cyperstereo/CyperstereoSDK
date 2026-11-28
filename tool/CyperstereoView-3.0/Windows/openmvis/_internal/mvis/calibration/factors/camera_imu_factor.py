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
"""IMU-propagated camera reprojection factor over the CameraCalib bundle.

Alternative to `CameraCalibInterpFactor`: a
frame stamped t is constrained by ONE anchor IMU state X_k (stamp t_k) and its
velocity/bias vector vb_k; the IMU pose at each corner's CAPTURE time is
obtained by integrating the raw base-IMU samples from t_k (`ImuSegment`):

    Δ_i      = dt_anchor + td + ρ_i·tr                  (dt_anchor = t − t_k, signed)
    w_I, a_I = corrected(w_m, a_m; b_g, b_a, Dw, Da, Tg)  (live values)
    D_R, D_v, D_p = ∫ over [t_k, t_k + Δ_i]              (ACI recurrence)
    R_i = R_k D_R,  p_i = p_k + v_k Δ_i + ½ g Δ_i² + R_k D_p
    T_CtoW = (R_i, p_i) · T_CtoI
    r_i    = π( T_CtoW⁻¹ T_BinW p_B,i ) − uv_i

so the time offsets and the rolling-shutter row times are handled by IMU
integration instead of the constant-twist chord between two states — the
corner poses follow the measured 200 Hz motion inside the state interval. The
velocity state re-enters the vision factor (through p_i), and so do the IMU
intrinsics (through the corrected measurements), which is deliberate: the
gyro scale Dw is then estimated jointly and consistently by both the IMU and
the vision terms. `gyro="raw"` (Dw = I, Da = I, Tg = 0; biases still removed)
and `dw_column=False` (Dw enters the residual but its Jacobian column is
zeroed) are diagnostic variants to measure that coupling.

    keys = [pose_k, vb_k, board, calib(ξ_C)] (+ imu_intr(ξ24) when calibrated)
           (+ δ_f, a per-frame capture-time offset, when frame_dt=True)

With frame_dt=True every frame carries its own scalar time offset δ_f added to
all its capture times (Δ_i = dt_anchor + td + δ_f + ρ_i·tr), given a zero-mean
prior by the pipeline (solve.vision_imu_frame_dt_sigma): it absorbs per-frame
capture-time jitter of the camera (the ELP's ~4 ms rms, E8 follow-up) that
otherwise biases the gyro scale through errors-in-variables. Its Jacobian is
the td column.

Analytic Jacobians (right tangent of T_i; T_Δ = (D_R, D_p)):

    X_k      : J_T · Ad(T_Δ⁻¹) = J_T · [[D_Rᵀ, 0], [−D_Rᵀ[D_p]×, D_Rᵀ]]
    v_k      : J_T[:, 3:6] · R_iᵀ Δ_i
    b_g, b_a : J_T · [[J_R_b], [D_Rᵀ J_p_b]]        (ACI bias Jacobians at the live bias)
    ξ24      : J_T · [[J_R_in], [D_Rᵀ J_p_in]]      (zero for gyro="raw" / dw_column=False)
    td       : J_T · [Jr(w̄δ)·w_I(t_i) ; R_iᵀ v_W(t_i)],  v_W = v_k + gΔ_i + R_k D_v
    tr       : ρ_i × td column
    board, T_CtoI, intrinsics : as in the interpolated factor

The position part of the td column treats the partial-step acceleration as
constant (exact for the rotation part); FD agreement on td/tr is therefore to
~1e-3 relative on shaking motion rather than machine precision. The gyro<->accel
offset t_ga (`tga=True`, trailing key) enters at its live value through ACI's
first-order model, in `ImuSegment`. Its rotation effect is identically zero under
dw_da, and its position effect is ~0.04 px over 50 ms; leaving it out pulled the
estimate by about 1 ms on the GS twin.
"""

from __future__ import annotations

import numpy as np
import gtsam

from .camera_calib import CameraCalib
from .imu_segment import ImuSegment
from ._calib_memo import get_calib_setup
from .corner_reproj import CornerReprojMixin
from .lie_groups import exp_so3, skew
from .numdiff import numerical_jacobian
from mvis.imu import create_imu_model


TD_REFRESH_S = 0.001


class _CorrectedProp:
    """A segment.Propagated moved to nearby inputs to first order; the same Jacobians."""
    __slots__ = ("D_R", "D_v", "D_p", "J_R_b", "J_p_b", "J_R_in", "J_p_in",
                 "J_R_tga", "J_p_tga", "dtheta_dDelta", "w_inI")

    def __init__(self, P, dd, db, dxi, dt_ga):
        self.J_R_b, self.J_p_b = P.J_R_b, P.J_p_b
        self.J_R_in, self.J_p_in = P.J_R_in, P.J_p_in
        self.J_R_tga, self.J_p_tga = P.J_R_tga, P.J_p_tga
        self.dtheta_dDelta, self.w_inI = P.dtheta_dDelta, P.w_inI
        th = P.J_R_b @ db + P.J_R_tga * dt_ga + P.dtheta_dDelta * dd
        dp = P.J_p_b @ db + P.J_p_tga * dt_ga + P.D_v * dd
        if dxi is not None:
            th = th + P.J_R_in @ dxi
            dp = dp + P.J_p_in @ dxi
        self.D_R = P.D_R @ exp_so3(th)
        self.D_p = P.D_p + dp
        self.D_v = P.D_v


class CameraCalibImuFactor(CornerReprojMixin):
    """Per-frame stacked reprojection over [pose_k, vb_k, board, calib(ξ), (ξ_imu)]."""

    def __init__(self, uv_meas, p_FinB, dt_anchor: float, segment: ImuSegment,
                 t_anchor: float, g_inW, calib_base: CameraCalib, image_size,
                 rolling_mode: int = 0, gyro: str = "calibrated",
                 dw_column: bool = True, frame_dt: bool = False,
                 board_pids=None, n_board_points: int = 0, tga: bool = False,
                 linearized: bool = False):
        self.uv = np.asarray(uv_meas, float).reshape(-1, 2)
        self.p_B = np.asarray(p_FinB, float).reshape(-1, 3)
        assert len(self.uv) == len(self.p_B), "one board corner per measurement"
        self.tga = bool(tga)
        # True: integrate once and correct linearly, like an IMU factor, instead of re-integrating
        # at every evaluation. See _linearized_props.
        self.linearized = bool(linearized)
        self._lin = None
        # Estimated board shape (optional, as in CameraCalibInterpFactor): one
        # height offset per board point in a trailing key, added along the
        # board normal at the corners' ids.
        self.pids = None if board_pids is None else np.asarray(board_pids, int).reshape(-1)
        self.n_bp = int(n_board_points)
        if self.pids is not None:
            assert len(self.pids) == len(self.uv) and self.n_bp > int(self.pids.max())
        assert gyro in ("calibrated", "raw")
        assert not (self.tga and gyro == "raw"), "t_ga needs the calibrated gyro"
        self.dt_anchor = float(dt_anchor)
        self.seg = segment
        self.t_anchor = float(t_anchor)
        self.g = np.asarray(g_inW, float)
        self.base = calib_base
        self.w_px, self.h_px = int(image_size[0]), int(image_size[1])
        self.rolling_mode = int(rolling_mode)
        self.gyro = gyro
        self.dw_column = bool(dw_column)
        self.frame_dt = bool(frame_dt)
        self.n = len(self.uv)
        self.rho = self._row_offsets()
        self.imu_base = create_imu_model(segment.model)   # ξ24 base (identity)
        self.keys = None                                   # set by make_factor
        self.key_dt = None

    # ------------------------------------------------------------------
    @property
    def uses_imu_key(self) -> bool:
        return self.gyro == "calibrated"

    @property
    def uses_heights_key(self) -> bool:
        return self.pids is not None

    def _intr(self, xi_imu):
        if not self.uses_imu_key or xi_imu is None:
            return None                                    # identity intrinsics
        return self.imu_base.retract(np.asarray(xi_imu, float))

    def deltas(self, calib: CameraCalib, dt_f=None) -> np.ndarray:
        """Signed capture offsets from the anchor state per corner (s)."""
        d_f = 0.0 if dt_f is None else float(np.asarray(dt_f).ravel()[0])
        return self.dt_anchor + calib.td + d_f + self.rho * calib.tr

    def _tga_value(self, t_ga):
        """Live t_ga for the propagation; None when this factor does not use it."""
        if not self.tga:
            return None
        # What the segment's accel does not include yet (t_lin = 0 until tga_relin.py moves it).
        return 0.0 if t_ga is None else float(np.asarray(t_ga).ravel()[0]) - self.seg.t_lin

    def _linearized_props(self, dl, b_g, b_a, xi_imu, tga, jac):
        """Propagation from a stored point plus first-order corrections, as in an IMU factor.

        The point is integrated again when the segment's accel moved (t_ga, see tga_relin.py), or,
        at a linearization, when a capture offset left TD_REFRESH_S around it: past that the window
        crosses IMU samples the stored rate no longer describes."""
        lin, seg = self._lin, self.seg
        xi = None if xi_imu is None else np.asarray(xi_imu, float)
        stale = lin is None or lin["version"] != seg.version
        if not stale and jac and seg.allow_td_refresh:
            stale = float(np.max(np.abs(dl - lin["dl"]))) > TD_REFRESH_S
        if stale:
            lin = self._lin = dict(
                props=seg.propagate(self.t_anchor, dl, b_g, b_a, self._intr(xi), jac=True, t_ga=tga),
                dl=np.array(dl), b=np.concatenate([b_g, b_a]), xi=xi, tga=tga, version=seg.version)
        db = np.concatenate([b_g, b_a]) - lin["b"]
        dxi = None if xi is None or lin["xi"] is None else xi - lin["xi"]
        dt_ga = 0.0 if tga is None else tga - lin["tga"]
        out, seen = [], {}
        for P0, d0, d in zip(lin["props"], lin["dl"], dl):
            key = (id(P0), d)               # a global-shutter frame has one capture time
            P = seen.get(key)
            if P is None:
                P = seen[key] = _CorrectedProp(P0, d - d0, db, dxi, dt_ga)
            out.append(P)
        return out

    def _poses(self, pose_k, vb, calib, xi_imu, jac, dt_f=None, t_ga=None):
        """Per-corner (T_i, Propagated, Δ_i) plus R_k, p_k, v_k."""
        R_k = pose_k.rotation().matrix()
        p_k = np.asarray(pose_k.translation(), float)
        vb = np.asarray(vb, float)
        v_k, b_g, b_a = vb[0:3], vb[3:6], vb[6:9]
        dl = self.deltas(calib, dt_f)
        if self.linearized:
            props = self._linearized_props(dl, b_g, b_a, xi_imu, self._tga_value(t_ga), jac)
        else:
            props = self.seg.propagate(self.t_anchor, dl, b_g, b_a,
                                       self._intr(xi_imu), jac=jac,
                                       t_ga=self._tga_value(t_ga))
        poses = []
        for i in range(self.n):
            P, d = props[i], dl[i]
            R_i = R_k @ P.D_R
            p_i = p_k + v_k * d + 0.5 * self.g * d * d + R_k @ P.D_p
            poses.append(gtsam.Pose3(gtsam.Rot3(R_i), gtsam.Point3(*p_i)))
        return poses, props, dl, R_k, v_k

    def error(self, pose_k, vb, T_BinW, xi, xi_imu=None, dt_f=None, h=None, t_ga=None):
        """Stacked 2N reprojection residual (unwhitened)."""
        return self._residuals(pose_k, vb, T_BinW, xi, xi_imu, dt_f, h, t_ga)[0]

    def _residuals(self, pose_k, vb, T_BinW, xi, xi_imu=None, dt_f=None, h=None, t_ga=None):
        setup = get_calib_setup(self.base, xi)      # shared memo (read-only)
        calib, proj = setup.calib, setup.proj
        poses, _p, _d, _R, _v = self._poses(pose_k, vb, calib, xi_imu, jac=False,
                                            dt_f=dt_f, t_ga=t_ga)
        err = np.zeros(2 * self.n)
        valid = np.zeros(self.n, bool)
        pB = self._pB(h)
        for i in range(self.n):
            T_CtoW = poses[i].compose(calib.T)
            p_W = T_BinW.transformFrom(gtsam.Point3(*pB[i]))
            try:
                px, _, _, _ = proj.project(T_CtoW, p_W, jac=False)
            except RuntimeError:                # cheirality / model range
                continue
            err[2 * i:2 * i + 2] = px - self.uv[i]
            valid[i] = True
        return err, valid

    def args_from_values(self, values: gtsam.Values):
        """(pose_k, vb, board, xi, xi_imu|None, dt_f|None, h|None) at a gtsam
        Values, using the keys from make_factor."""
        k = self.keys
        q = 4
        xi_imu = dt_f = h = t_ga = None
        if self.uses_imu_key:
            xi_imu = values.atVector(k[q]); q += 1
        if self.frame_dt:
            dt_f = values.atVector(k[q]); q += 1
        if self.uses_heights_key:
            h = values.atVector(k[q]); q += 1
        if self.tga:
            t_ga = values.atVector(k[q])
        return (values.atPose3(k[0]), values.atVector(k[1]), values.atPose3(k[2]),
                values.atVector(k[3]), xi_imu, dt_f, h, t_ga)

    def camera_points_values(self, values: gtsam.Values) -> np.ndarray:
        """Each corner in the camera frame at a gtsam Values (N x 3), by the same
        chain as `_residuals`; for the corner accounting, which needs the angle."""
        pose_k, vb, T_BinW, xi, xi_imu, dt_f, h, t_ga = self.args_from_values(values)
        calib = get_calib_setup(self.base, xi).calib
        poses = self._poses(pose_k, vb, calib, xi_imu, jac=False, dt_f=dt_f, t_ga=t_ga)[0]
        pB = self._pB(h)
        out = np.empty((self.n, 3))
        for i in range(self.n):
            p_W = T_BinW.transformFrom(gtsam.Point3(*pB[i]))
            out[i] = np.asarray(poses[i].compose(calib.T).transformTo(p_W), float)
        return out

    # ------------------------------------------------------------------
    def error_and_jacobians(self, pose_k, vb, T_BinW, xi, xi_imu=None, dt_f=None,
                            h=None, t_ga=None):
        """(error 2N, [H_pose 2N×6, H_vb 2N×9, H_board 2N×6,
        H_calib 2N×DIM, (H_imu 2N×24 when calibrated), (H_dt 2N×1 when frame_dt),
        (H_heights 2N×n_board_points when the board shape is estimated),
        (H_tga 2N×1 when tga)])."""
        setup = get_calib_setup(self.base, xi)      # shared memo (read-only)
        calib, proj = setup.calib, setup.proj
        n_intr = calib.n_intr
        R_B = T_BinW.rotation().matrix()
        T_C = calib.T
        Ad_TCinv = setup.Ad_TCinv
        Jr_xiT = setup.Jr_xiT

        poses, props, dl, R_k, v_k = self._poses(pose_k, vb, calib, xi_imu, jac=True,
                                                 dt_f=dt_f, t_ga=t_ga)

        err = np.zeros(2 * self.n)
        H_pose = np.zeros((2 * self.n, 6))
        H_vb = np.zeros((2 * self.n, 9))
        H_board = np.zeros((2 * self.n, 6))
        H_calib = np.zeros((2 * self.n, calib.DIM))
        H_imu = np.zeros((2 * self.n, 24)) if self.uses_imu_key else None
        H_dt = np.zeros((2 * self.n, 1)) if self.frame_dt else None
        H_h = np.zeros((2 * self.n, self.n_bp)) if self.uses_heights_key else None
        H_tga = np.zeros((2 * self.n, 1)) if self.tga else None
        pB = self._pB(h)
        n_B = R_B[:, 2]                         # d p_W / d h  (board normal in W)

        for i in range(self.n):
            P, d, T_I = props[i], dl[i], poses[i]
            T_CtoW = T_I.compose(T_C)
            p_W = np.asarray(T_BinW.transformFrom(gtsam.Point3(*pB[i])), float)
            try:
                px, J_pose, J_point, J_intr = proj.project(T_CtoW, p_W)
            except RuntimeError:                # cheirality: zero rows
                continue
            err[2 * i:2 * i + 2] = px - self.uv[i]
            r = slice(2 * i, 2 * i + 2)
            J_T = J_pose @ Ad_TCinv            # w.r.t. T_i right tangent
            DRt = P.D_R.T
            R_i = T_I.rotation().matrix()

            # anchor state: Ad(T_Δ⁻¹)
            M = np.zeros((6, 6))
            M[0:3, 0:3] = DRt
            M[3:6, 0:3] = -DRt @ skew(P.D_p)
            M[3:6, 3:6] = DRt
            H_pose[r] = J_T @ M

            # velocity (world) and biases
            H_vb[r, 0:3] = J_T[:, 3:6] @ (R_i.T * d)
            H_vb[r, 3:9] = J_T @ np.vstack([P.J_R_b, DRt @ P.J_p_b])

            if H_imu is not None and self.dw_column:
                H_imu[r] = J_T @ np.vstack([P.J_R_in, DRt @ P.J_p_in])

            if H_tga is not None:
                H_tga[r, 0] = J_T @ np.concatenate([P.J_R_tga, DRt @ P.J_p_tga])

            H_board[r, 0:3] = J_point @ (-R_B @ skew(pB[i]))
            H_board[r, 3:6] = J_point @ R_B
            if H_h is not None:
                H_h[r, self.pids[i]] = J_point @ n_B
            H_calib[r, 0:6] = J_pose @ Jr_xiT
            v_W = v_k + self.g * d + R_k @ P.D_v
            zeta_t = np.concatenate([P.dtheta_dDelta, R_i.T @ v_W])
            col_t = J_T @ zeta_t
            H_calib[r, 6] = col_t
            if H_dt is not None:
                H_dt[r, 0] = col_t                  # δ_f enters exactly like td
            H_calib[r, 7] = self.rho[i] * col_t
            H_calib[r, 8:8 + n_intr] = J_intr

        Hs = [H_pose, H_vb, H_board, H_calib]
        if H_imu is not None:
            Hs.append(H_imu)
        if H_dt is not None:
            Hs.append(H_dt)
        if H_h is not None:
            Hs.append(H_h)
        if H_tga is not None:
            Hs.append(H_tga)
        return err, Hs

    def _numerical_jacobians(self, pose_k, vb, T_BinW, xi, xi_imu=None,
                             dt_f=None, h=None, t_ga=None, eps: float = 1e-6):
        args = [pose_k, vb, T_BinW, xi]
        if self.uses_imu_key:
            args.append(np.zeros(24) if xi_imu is None else np.asarray(xi_imu, float))
        if self.frame_dt:
            args.append(np.zeros(1) if dt_f is None else np.asarray(dt_f, float).reshape(1))
        if self.uses_heights_key:
            args.append(np.zeros(self.n_bp) if h is None else np.asarray(h, float))
        if self.tga:
            args.append(np.zeros(1) if t_ga is None else np.asarray(t_ga, float).reshape(1))

        def fn(*a):
            a = list(a); q = 4
            x_imu = d_f = hh = tg = None
            if self.uses_imu_key:
                x_imu = a[q]; q += 1
            if self.frame_dt:
                d_f = a[q]; q += 1
            if self.uses_heights_key:
                hh = a[q]; q += 1
            if self.tga:
                tg = a[q]
            return self.error(a[0], a[1], a[2], a[3], x_imu, d_f, hh, tg)
        _, blocks = numerical_jacobian(fn, args, eps)
        return blocks

    # ------------------------------------------------------------------
    def make_factor(self, k_pose, k_vb, k_board, k_calib, k_imu=None, k_dt=None,
                    k_heights=None, sigma_px: float = 1.0, huber_px: float = 0.0,
                    k_tga=None):
        """4- to 8-key gtsam.CustomFactor (isotropic pixel noise, optional Huber)."""
        ks = [k_pose, k_vb, k_board, k_calib]
        if self.uses_imu_key:
            assert k_imu is not None, "calibrated gyro needs the IMU-intrinsics key"
            ks.append(k_imu)
        if self.frame_dt:
            assert k_dt is not None, "frame_dt needs the per-frame time-offset key"
            ks.append(k_dt)
        self.key_dt = k_dt if self.frame_dt else None   # explicit: heights may follow
        if self.uses_heights_key:
            assert k_heights is not None, "board_pids needs the board-heights key"
            ks.append(k_heights)
        if self.tga:
            assert k_tga is not None, "tga needs the t_ga key"
            ks.append(k_tga)
        self.keys = list(ks)
        keys = gtsam.KeyVector(ks)
        noise = gtsam.noiseModel.Isotropic.Sigma(2 * self.n, sigma_px)
        if huber_px > 0:
            noise = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(huber_px / sigma_px),
                noise)
        n_keys = len(ks)

        def err_func(this, values, H):
            args = self.args_from_values(values)
            if H is None:
                return self.error(*args)
            e, Js = self.error_and_jacobians(*args)
            for i in range(n_keys):
                H[i] = Js[i]
            return e

        return gtsam.CustomFactor(noise, keys, err_func)
