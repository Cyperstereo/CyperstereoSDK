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
"""ImuSegment — live IMU propagation from a pose state to arbitrary nearby times.

The IMU-propagated vision factor (`camera_imu_factor.py`) needs, for every
corner of a frame, the IMU pose at the corner's CAPTURE time
t_k + Δ_i (Δ_i = dt_frame + td + ρ_i·tr, signed) given the state at t_k. This
helper integrates the raw base-IMU samples over [t_k, t_k + Δ_i] with the same
ACI recurrence as `aci.ACIPreintegrator.feed_imu_single` (avg_meas, corrected
measurements, bias + intrinsic Jacobians) but WITHOUT the covariance, at the
LIVE biases and intrinsics — the factor is re-evaluated at every linearisation,
so no first-order bias correction is needed. With `t_ga` given, the accel takes
ACI's first-order gyro<->accel offset (a + t_ga * a_dot, raw real-time slope over
the step) at the live value, and `J_*_tga` is accumulated.

Per target Δ the output is (right-tangent conventions, as in `aci.py`):

    R(t_k+Δ) = R_k · D_R
    p(t_k+Δ) = p_k + v_k·Δ + ½ g Δ² + R_k · D_p
    v(t_k+Δ) = v_k + g·Δ + R_k · D_v

for Δ of EITHER sign. A negative Δ is integrated in reversed time: the samples
are walked backwards from t_k with the CORRECTED body rate negated (the Tg
term makes negating the raw sample wrong) and the body acceleration unchanged;
the formulas above then hold with D_v = −D_v' of the reversed run.

Checkpoints sit at the anchor time and at every raw sample in between; the
step from the last checkpoint to the target uses the midpoint of the LINEARLY
INTERPOLATED signal, ½(w_m + w(t_m+δ)), which reproduces the full avg_meas step
exactly at δ = dt (the pose is continuous when td/tr move a target across a
sample) and makes d/dδ[w̄·δ] = w(t_m+δ), so the time-offset column of the factor
is the end-point body rate (through Jr of the partial step).
"""

from __future__ import annotations

import numpy as np

from mvis.imu import ImuModel, create_imu_model
from .aci import (integrate_R, integrate_RAJ, _IntrPack, intr_pack, _state_recurrence,
                  _phi_hwa, step_input_jacobians)



class _Corrected:
    """Corrected measurements at one time (see ImuModel.correct).

    `pack` is the `_IntrPack` of the intrinsics (an `ImuModel` is accepted and
    packed); `Ma = R_atoI @ Da` and `Mw = R_wtoI @ Dw` keep the original
    left-to-right grouping `(R_atoI @ Da) @ a_ina`."""
    __slots__ = ("a_ina", "a_inI", "w_inw", "w_inI")

    def __init__(self, pack: _IntrPack, b_g, b_a, w_m, a_m):
        pack = intr_pack(pack)
        self.a_ina = a_m - b_a
        self.a_inI = pack.Ma @ self.a_ina
        self.w_inw = w_m - b_g - pack.Tg @ self.a_inI
        self.w_inI = pack.Mw @ self.w_inw


class _State:
    """Preintegrated deltas + Jacobians (rows [θ, p, v]; cols [b_g, b_a] / ξ24)."""
    __slots__ = ("D_R", "D_v", "D_p", "J_b", "J_in", "J_tga", "Dt", "Jr_last")

    def __init__(self):
        self.D_R = np.eye(3)
        self.D_v = np.zeros(3)
        self.D_p = np.zeros(3)
        self.J_b = np.zeros((9, 6))
        self.J_in = np.zeros((9, 24))
        self.J_tga = np.zeros(9)
        self.Dt = 0.0
        self.Jr_last = np.eye(3)


class Propagated:
    """Result of one target: deltas in the ANCHOR frame + Jacobian blocks.

    D_v already carries the direction sign (v = v_k + gΔ + R_k D_v for either
    sign of Δ). `dtheta_dDelta` is d(right-tangent θ of R)/dΔ at the target —
    the end-point body rate through Jr of the partial step; the position part
    of the time derivative is v(t_k+Δ) (assembled by the factor)."""
    __slots__ = ("D_R", "D_v", "D_p", "J_R_b", "J_p_b", "J_R_in", "J_p_in",
                 "J_R_tga", "J_p_tga", "dtheta_dDelta", "w_inI")

    def __init__(self, st: _State, sign: int, w_inI_end: np.ndarray):
        self.D_R = st.D_R
        self.D_v = sign * st.D_v
        self.D_p = st.D_p
        self.J_R_b, self.J_p_b = st.J_b[0:3], st.J_b[3:6]
        self.J_R_in, self.J_p_in = st.J_in[0:3], st.J_in[3:6]
        self.J_R_tga, self.J_p_tga = st.J_tga[0:3], st.J_tga[3:6]
        self.w_inI = w_inI_end
        # forward: δθ/dΔ = Jr(w̄_eff δ) w_eff, w_eff = w_I; backward: w_eff = −w_I
        # and dδ/dΔ = −1 — the same expression either way.
        self.dtheta_dDelta = st.Jr_last @ w_inI_end


def _step(st: _State, d_t: float, c0: _Corrected, c1: _Corrected,
          pack: _IntrPack, sign: int, jac: bool, t_ga=None) -> _State:
    """One avg_meas ACI step over d_t from corrected endpoint measurements,
    with the body rate multiplied by `sign` (−1 = reversed time). Returns a
    new state; `st` is untouched. Mirrors `ACIPreintegrator.feed_imu_single`
    minus covariance / t_ga. `pack` is the `_IntrPack` of the intrinsics (an
    `ImuModel` is accepted and packed); jac=False takes the rotation-only
    `integrate_R` (the RAJ integrals are only used by the Jacobians). `t_ga`
    None = no gyro<->accel offset and no `J_tga` (the previous behaviour)."""
    a_ina = 0.5 * (c0.a_ina + c1.a_ina)
    a_hat = 0.5 * (c0.a_inI + c1.a_inI)
    w_inw = 0.5 * (c0.w_inw + c1.w_inw)
    w_hat = sign * 0.5 * (c0.w_inI + c1.w_inI)

    if t_ga is not None:
        # ACI's first-order offset: da/dt_ga = Ma @ a_dot, a_dot = raw slope in
        # REAL time; through Tg it also reaches the gyro (zero while Tg = 0).
        pack = intr_pack(pack)
        a_dot = sign * (c1.a_ina - c0.a_ina) / d_t
        da = pack.Ma @ a_dot
        dw_inw = -pack.Tg @ da
        a_ina = a_ina + t_ga * a_dot
        a_hat = a_hat + t_ga * da
        w_inw = w_inw + t_ga * dw_inw
        w_hat = w_hat + t_ga * sign * (pack.Mw @ dw_inw)

    if jac:
        dR, int_R, iint_R, Jr, int_RAJ, iint_RAJ = integrate_RAJ(d_t, w_hat, a_hat)
    else:
        dR, int_R, iint_R, Jr = integrate_R(d_t, w_hat)

    out = _State()
    R_prev = st.D_R
    out.D_R, out.D_v, out.D_p = _state_recurrence(st.D_R, st.D_v, st.D_p, d_t,
                                                  dR, int_R, iint_R, a_hat)
    out.Dt = st.Dt + d_t
    out.Jr_last = Jr
    if not jac:
        return out

    pack = intr_pack(pack)
    H_b, H_in = step_input_jacobians(pack, sign, a_ina, w_inw, w_hat, a_hat)

    phi_nn, H_wa = _phi_hwa(R_prev, dR, d_t, Jr, int_R, iint_R, int_RAJ, iint_RAJ, a_hat)

    out.J_b = phi_nn @ st.J_b + H_wa @ H_b
    out.J_in = phi_nn @ st.J_in + H_wa @ H_in
    if t_ga is not None:
        H_tga = np.concatenate([sign * (pack.Mw @ dw_inw), da])
        out.J_tga = phi_nn @ st.J_tga + H_wa @ H_tga
    return out


class ImuSegment:
    """Raw base-IMU record + propagation from an anchor time to signed offsets."""

    def __init__(self, t_imu, w_m, a_m, model: int):
        self.t = np.asarray(t_imu, float)
        self.w = np.asarray(w_m, float).reshape(-1, 3)
        self.a = np.asarray(a_m, float).reshape(-1, 3)
        assert len(self.t) == len(self.w) == len(self.a) and len(self.t) >= 2
        assert np.all(np.diff(self.t) > 0), "IMU stamps must be strictly increasing"
        self.model = int(model)
        self.identity = create_imu_model(self.model)
        self.t_lin = 0.0        # t_ga that self.a already includes (tga_relin.py)
        self.version = 0        # bumped when self.a changes: cached propagations are stale
        self.allow_td_refresh = True    # the optimizer holds this off in the first iterations

    def relinearize_tga(self, a_shifted, t_lin: float) -> None:
        """Use the accel resampled at t + t_lin; propagate() then takes t_ga - t_lin."""
        self.a = np.asarray(a_shifted, float).reshape(-1, 3)
        self.t_lin = float(t_lin)
        self.version += 1

    # ------------------------------------------------------------------
    def covers(self, t0: float, t1: float) -> bool:
        """True when [t0, t1] lies inside the IMU record (no clamping needed)."""
        return bool(self.t[0] <= min(t0, t1) and max(t0, t1) <= self.t[-1])

    def _interp(self, tq: float):
        """Linear interpolation of (w_m, a_m) at tq (clamped at the record ends)."""
        t = self.t
        i = int(np.searchsorted(t, tq, side="right")) - 1
        i = min(max(i, 0), len(t) - 2)
        lam = (tq - t[i]) / (t[i + 1] - t[i])
        lam = min(max(lam, 0.0), 1.0)
        return ((1.0 - lam) * self.w[i] + lam * self.w[i + 1],
                (1.0 - lam) * self.a[i] + lam * self.a[i + 1])

    # ------------------------------------------------------------------
    def propagate(self, t_anchor: float, deltas, b_g, b_a,
                  intr: ImuModel | None = None, jac: bool = True, t_ga=None):
        """Propagate from t_anchor to t_anchor + deltas[i] for every i.

        Returns a list of `Propagated` (one per delta, same order). `intr`
        None = identity intrinsics (raw gyro / accel, biases only)."""
        deltas = np.asarray(deltas, float)
        b_g = np.asarray(b_g, float)
        b_a = np.asarray(b_a, float)
        intr = self.identity if intr is None else intr
        pack = _IntrPack(intr)                 # products once per propagate
        out = [None] * len(deltas)
        for sign in (1, -1):
            idx = np.nonzero(deltas >= 0.0 if sign > 0 else deltas < 0.0)[0]
            if idx.size:
                self._propagate_dir(t_anchor, deltas, idx, sign, b_g, b_a,
                                    pack, jac, out, t_ga)
        return out

    def _propagate_dir(self, t_anchor, deltas, idx, sign, b_g, b_a, intr,
                       jac, out, t_ga=None):
        intr = intr_pack(intr)
        tau = sign * deltas[idx]                     # ≥ 0, "elapsed" time
        order = np.argsort(tau)
        tau_max = float(tau[order[-1]])
        t = self.t
        if sign > 0:
            inside = np.nonzero((t > t_anchor) & (t < t_anchor + tau_max))[0]
        else:
            inside = np.nonzero((t > t_anchor - tau_max) & (t < t_anchor))[0][::-1]
        # checkpoints: anchor (interpolated) then the raw samples walked in
        # the integration direction
        tau_cp = np.concatenate([[0.0], sign * (t[inside] - t_anchor)])
        w0, a0 = self._interp(t_anchor)
        meas = [_Corrected(intr, b_g, b_a, w0, a0)]
        meas += [_Corrected(intr, b_g, b_a, self.w[i], self.a[i]) for i in inside]
        states = [_State()]
        for m in range(1, len(tau_cp)):
            states.append(_step(states[m - 1], tau_cp[m] - tau_cp[m - 1],
                                meas[m - 1], meas[m], intr, sign, jac, t_ga))
        for q in order:
            tq = float(tau[q])
            m = int(np.searchsorted(tau_cp, tq, side="right")) - 1
            delta = tq - tau_cp[m]
            if delta < 1e-12:
                # exactly on a checkpoint: the partial step has δ -> 0, so its
                # Jr -> I (the time derivative is continuous across checkpoints)
                st = states[m]
                st_end = _State()
                st_end.D_R, st_end.D_v, st_end.D_p = st.D_R, st.D_v, st.D_p
                st_end.J_b, st_end.J_in, st_end.Dt = st.J_b, st.J_in, st.Dt
                st_end.J_tga = st.J_tga
                out[idx[q]] = Propagated(st_end, sign, meas[m].w_inI)
                continue
            wq, aq = self._interp(t_anchor + sign * tq)
            cq = _Corrected(intr, b_g, b_a, wq, aq)
            st = _step(states[m], delta, meas[m], cq, intr, sign, jac, t_ga)
            out[idx[q]] = Propagated(st, sign, cq.w_inI)
