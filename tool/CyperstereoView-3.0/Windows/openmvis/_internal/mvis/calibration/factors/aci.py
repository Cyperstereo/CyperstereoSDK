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
"""ACI (Analytic Combined IMU) preintegration with intrinsics.

Pure-Python port of the C++ MVIS `aci/ACIBase` + `aci/ACI_In` (`feed_IMU_single`,
`integrate_RAJ`). Integrates raw IMU (w_m, a_m) over an interval into the
preintegrated deltas (D_R, D_v, D_p) plus the bias and intrinsic Jacobians and
the 15x15 measurement covariance that the `ImuFactorACI_In_SE3` factor consumes.

Papers: ACI2, Yang et al., "Analytic Combined IMU Integration (ACI2) For Visual Inertial Navigation", ICRA 2020,
doi 10.1109/ICRA40945.2020.9197280; ACI3 (with intrinsics), Yang, Geneva and Huang, IJRR 2024.
Reference: ACI tech report — eq. 74–84 (integration), 125–134 (bias Jacobians),
141–154 (covariance). Validated against the C++ ACI_In oracle.

Measurement-correction model (one step):
    a_ina = a_m - b_a
    a_inI = R_atoI · Da · a_ina
    w_inw = w_m - b_g - Tg · a_inI
    w_inI = R_wtoI · Dw · w_inw
With avg_meas the midpoint of (start, end) corrected w/a is used.

Intrinsic param order (24, model-dependent — see ImuModel):
    Kalibr/RPNG: [x_Dw(6), x_Da(6), x_Tg(9), R(3)]
    Calyx:       [x_Dw(9), x_Da(6), x_Tg(9)]   (full Dw, no rotation DOF)
"""

from __future__ import annotations

import math

import numpy as np

from mvis.imu import (
    ImuModel, KALIBR, RPNG, n_dw_for, dw_jacobian, da_jacobian, tg_jacobian,
    create_imu_model,
)
from mvis.calibration.factors.lie_groups import skew, so3_left_jacobian

_I3 = np.eye(3)
_Z3 = np.zeros((3, 3))
_SMALL = (1.0 / 180.0) * np.pi / 2.0  # 0.5 deg threshold (matches C++)


def _rot_integrals(d_t: float, w_hat: np.ndarray):
    """Rotation-only part of one ACI step (a speed-up): the original
    `integrate_RAJ` lines that do not involve `a_hat`, verbatim and in the
    same order, with the same three branches (`k_hat = 0` when |ω| < 1e-8,
    `small_w`, `small_th`).

    Returns (R, xi_1, xi_2, J) plus the intermediates `integrate_RAJ` needs
    for xi_3 / xi_4: (k_hat, sK, sK2, cos_dth, sin_dth, d_th, w_norm, small_w).

    Scalar math: `math.sqrt(w_hat @ w_hat)` / `math.cos` / `math.sin` replace
    `float(np.linalg.norm)` / `np.cos` / `np.sin`. numpy 1.26's `norm` with
    axis=None is `sqrt(x.dot(x))` and its float64 sin/cos take the libm loop
    on this (AVX2) build, so the values are bitwise the same — verified by
    `tests/cam_imu/test_aci.py::test_integrate_r_matches_oracle_10k` against the
    frozen oracle on 10k inputs incl. the branch edges (keep only while
    `array_equal` holds; SVML builds may differ by 1 ulp).
    """
    w_norm = math.sqrt(w_hat @ w_hat)
    d_th = w_norm * d_t
    k_hat = np.zeros(3) if w_norm < 1e-8 else w_hat / w_norm

    d_t2 = d_t ** 2
    w2 = w_norm ** 2
    cos_dth, sin_dth = math.cos(d_th), math.sin(d_th)
    d_th2, d_th3 = d_th ** 2, d_th ** 3
    sK = skew(k_hat)
    sK2 = sK @ sK

    small_w = w_norm < _SMALL
    small_th = d_th < _SMALL

    R = _I3 + sin_dth * sK + (1.0 - cos_dth) * sK2
    if not small_th:
        J = _I3 - (1.0 - cos_dth) / d_th * sK + (d_th - sin_dth) / d_th * sK2
    else:
        # Taylor series of the same Jr(φ): (1-cos θ)/θ ≈ θ/2 - θ³/24 and
        # (θ-sin θ)/θ ≈ θ²/6 - θ⁴/120. (The C++ port's `I - sin θ [k]× + ...`
        # is I - [φ]× ..., twice the true first-order term — a 0.5 % Jacobian
        # error at 200 Hz that only the live-Jacobian factors notice.)
        J = (_I3 - (0.5 * d_th - d_th3 / 24.0) * sK
             + (d_th2 / 6.0 - d_th2 * d_th2 / 120.0) * sK2)

    if not small_w:
        xi_1 = _I3 * d_t + (1.0 - cos_dth) / w_norm * sK + (d_t - sin_dth / w_norm) * sK2
        xi_2 = (0.5 * d_t2 * _I3
                + (d_th - sin_dth) / w2 * sK
                + (0.5 * d_t2 - (1.0 - cos_dth) / w2) * sK2)
    else:
        xi_1 = d_t * (_I3 + sin_dth * sK + (1.0 - cos_dth) * sK2)
        xi_2 = 0.5 * d_t * xi_1

    return R, xi_1, xi_2, J, (k_hat, sK, sK2, cos_dth, sin_dth, d_th, w_norm, small_w)


def integrate_R(d_t: float, w_hat: np.ndarray):
    """Rotation-only closed-form integrals over one step (a speed-up).

    Returns (R, xi_1, xi_2, J) — bitwise `integrate_RAJ(d_t, w_hat, a)[:4]`
    for any `a` (same lines, same branches). `J` (the right Jacobian of the
    step) is always returned: `imu_segment._step` stores it as `Jr_last` and
    `Propagated` needs it for `dtheta_dDelta` also when jac=False.
    """
    R, xi_1, xi_2, J, _ = _rot_integrals(d_t, w_hat)
    return R, xi_1, xi_2, J


def integrate_RAJ(d_t: float, w_hat: np.ndarray, a_hat: np.ndarray):
    """Closed-form rotation + RAJ integrals over one step.

    Returns (R, xi_1, xi_2, J, xi_3, xi_4):
      R     = Exp(w_hat·d_t)            (incremental rotation dR)
      xi_1  = ∫ R dt        (int_R)
      xi_2  = ∫∫ R dt       (iint_R)
      J     = right Jacobian Jr(w_hat·d_t)
      xi_3  = ∫ R [a]× dt   (int_RAJ)
      xi_4  = ∫∫ R [a]× dt  (iint_RAJ)

    The rotation part comes from `_rot_integrals` (= `integrate_R`); xi_3 /
    xi_4 keep their original expressions verbatim. The scalar-prefixed products
    (`c / w2 * sA @ sK` parses as `((c / w2) * sA) @ sK`) are deliberately NOT
    factored: there is no shared product, and a reassociation is not bitwise.
    """
    R, xi_1, xi_2, J, (k_hat, sK, sK2, cos_dth, sin_dth, d_th, w_norm, small_w) = \
        _rot_integrals(d_t, w_hat)

    d_t2, d_t3 = d_t ** 2, d_t ** 3
    w2, w3 = w_norm ** 2, w_norm ** 3
    d_th2, d_th3 = d_th ** 2, d_th ** 3
    sA = skew(a_hat)
    kdota = float(k_hat @ a_hat)

    if not small_w:
        xi_3 = (0.5 * d_t2 * sA
                + (sin_dth - d_th) / w2 * sA @ sK
                + (sin_dth - d_th * cos_dth) / w2 * sK @ sA
                + (0.5 * d_t2 - (1.0 - cos_dth) / w2) * sA @ sK2
                + (0.5 * d_t2 + (1.0 - cos_dth - d_th * sin_dth) / w2) * (sK2 @ sA + kdota * sK)
                - (3 * sin_dth - 2 * d_th - d_th * cos_dth) / w2 * kdota * sK2)
        xi_4 = (1.0 / 6 * d_t3 * sA
                + (2 * (1.0 - cos_dth) - d_th2) / (2 * w3) * sA @ sK
                + (2 * (1.0 - cos_dth) - d_th * sin_dth) / w3 * sK @ sA
                + ((sin_dth - d_th) / w3 + d_t3 / 6) * sA @ sK2
                + ((d_th - 2 * sin_dth + 1.0 / 6 * d_th3 + d_th * cos_dth) / w3) * (sK2 @ sA + kdota * sK)
                + (4 * cos_dth - 4 + d_th2 + d_th * sin_dth) / w3 * kdota * sK2)
    else:
        xi_3 = 0.5 * d_t2 * (sA
                             + sin_dth * (-sA @ sK + sK @ sA + kdota * sK2)
                             + (1.0 - cos_dth) * (sA @ sK2 + sK2 @ sA + kdota * sK))
        xi_4 = 1.0 / 3 * d_t * xi_3

    return R, xi_1, xi_2, J, xi_3, xi_4


class _IntrPack:
    """Intrinsic-model products built ONCE per `ImuSegment.propagate` /
    `ACIPreintegrator.feed_imu_interval` instead of per step (a speed-up).

    Every entry is the exact left-prefix product that one call site evaluated
    inline before, with Python's left-to-right grouping kept (`A @ B @ C @ v`
    is `((A @ B) @ C) @ v`), so each site's result is bitwise unchanged. The
    sites and their prefixes (see the frozen copies in `tests/_reference/`):

    `imu_segment._Corrected` / `feed_imu_single` corrections / the endpoint
    caches `a_m_s, a_m_e, w_m_s, w_m_e` of `feed_imu_interval`:
        Ma = R_atoI @ Da            a_inI = Ma @ a_ina
        Mw = R_wtoI @ Dw            w_inI = Mw @ w_inw;  w_inw uses Tg @ a_inI
    `feed_imu_single` t_ga row:
        H_tga        = nMwTg @ da_inI_tga  with nMwTg = (-Mw) @ Tg, da_inI_tga = Ma @ a_dot
    `step_input_jacobians` (sign s = +1 for `feed_imu_single`, -1 for the
    reversed-time walk of `imu_segment._step`; both take these, so
    the forward H_b[0:3,3:6] is ((Mw @ Tg) @ Ma), no longer
    ((((R_wtoI @ Dw) @ Tg) @ R_atoI) @ Da), a rounding-level change that is
    exactly zero while Tg = 0):
        Hb[s][0:3,0:3] = (-s) * Mw;  Hb[s][0:3,3:6] = ((s * Mw) @ Tg) @ Ma;  Hb[s][3:6,3:6] = -Ma
        H_in dw      = sRw[s] @ dwj(w_inw)          sRw[s]     = s * R_wtoI
        H_in da      = nsMwTgRa[s] @ daj            nsMwTgRa[s] = (((-s) * Mw) @ Tg) @ R_atoI
                       R_atoI @ daj
        H_in tg      = nsMw[s] @ tgj(a_hat)         nsMw[s]    = (-s) * Mw
        H_in r       = (-skew(w_hat)) @ R_wtoI                        (KALIBR)
                       (sMwTg[s] @ skew(a_hat)) @ R_atoI, (-skew(a_hat)) @ R_atoI (RPNG)
                                                    sMwTg[s]   = (s * Mw) @ Tg
    `daj` (`da_jacobian(a_ina)`) is evaluated once per step and reused for
    both rows (it was computed twice with identical arguments).

    The per-model parameter Jacobians are bound from `type(intr)` so the hot
    path skips the `dw_jacobian(w, model)` registry dispatch. The pack
    captures the model's values at build time — build it inside the call
    that uses it (the intrinsics are retracted anew per factor evaluation).
    """
    __slots__ = ("model", "n_dw", "da_off", "tg_off", "r_off", "has_R",
                 "R_atoI", "R_wtoI", "Tg", "Ma", "Mw",
                 "nMwTg", "Hb", "sRw", "nsMw", "nsMwTgRa", "sMwTg",
                 "dwj", "daj", "tgj")

    def __init__(self, intr: ImuModel):
        self.model = int(intr.model)
        self.n_dw, self.da_off, self.tg_off = intr.n_dw, intr.da_off, intr.tg_off
        self.has_R = bool(intr.has_R)
        self.r_off = intr.r_off if self.has_R else None
        R_atoI = intr.R_atoI
        R_wtoI = intr.R_wtoI
        Dw = intr.Dw()
        Da = intr.Da()
        Tg = intr.Tg()
        self.R_atoI, self.R_wtoI, self.Tg = R_atoI, R_wtoI, Tg
        Ma = R_atoI @ Da
        Mw = R_wtoI @ Dw
        self.Ma, self.Mw = Ma, Mw
        self.nMwTg = -Mw @ Tg                 # feed_imu_single's t_ga row
        # imu_segment._step prefixes, per sign
        self.Hb, self.sRw, self.nsMw, self.nsMwTgRa, self.sMwTg = {}, {}, {}, {}, {}
        for sign in (1, -1):
            H_b = np.zeros((6, 6))
            H_b[0:3, 0:3] = -sign * Mw
            H_b[0:3, 3:6] = sign * Mw @ Tg @ Ma
            H_b[3:6, 3:6] = -Ma
            self.Hb[sign] = H_b
            self.sRw[sign] = sign * R_wtoI
            self.nsMw[sign] = -sign * Mw
            self.nsMwTgRa[sign] = -sign * Mw @ Tg @ R_atoI
            self.sMwTg[sign] = sign * Mw @ Tg
        cls = type(intr)
        dwj = getattr(cls, "dw_param_jacobian", None)
        daj = getattr(cls, "da_param_jacobian", None)
        model = self.model
        self.dwj = dwj if dwj is not None else (lambda w: dw_jacobian(w, model))
        self.daj = daj if daj is not None else (lambda a: da_jacobian(a, model))
        self.tgj = tg_jacobian


def intr_pack(intr) -> _IntrPack:
    """`intr` as an `_IntrPack` (an `ImuModel` is packed; a pack passes through)."""
    return intr if isinstance(intr, _IntrPack) else _IntrPack(intr)


def correct_measurements(pack: _IntrPack, b_g, b_a, w_m_0, a_m_0, w_m_1, a_m_1,
                         avg_meas: bool):
    """Measurement correction (intrinsics + bias) of BOTH endpoints of one step
    and the `avg_meas` midpoint, as `feed_imu_single` evaluates it (speed-up
    The one helper shared by `feed_imu_single` and the state-only walk of
    `pipeline_cam_imu._densify_velocities`).

    Returns (a_ina, a_hat, w_inw, w_hat): the bias-corrected accel in the accel
    frame, the corrected accel / gyro in the IMU frame and the corrected gyro
    before `Mw` — the four quantities the step consumes. With `avg_meas` each
    is the `(x0 + x1) / 2.0` average of the endpoint corrections; without it
    the start-point correction. Both endpoints are re-corrected on every call
    (no caching across steps), so a caller that walks consecutive samples gets
    bitwise what `feed_imu_interval` gives.
    """
    Tg = pack.Tg
    Ma, Mw = pack.Ma, pack.Mw
    a0_ina = a_m_0 - b_a
    a1_ina = a_m_1 - b_a
    a0_inI = Ma @ a0_ina
    a1_inI = Ma @ a1_ina
    w0_inw = w_m_0 - b_g - Tg @ a0_inI
    w0_inI = Mw @ w0_inw
    w1_inw = w_m_1 - b_g - Tg @ a1_inI
    w1_inI = Mw @ w1_inw

    a_ina, a_hat = a0_ina, a0_inI
    w_inw, w_hat = w0_inw, w0_inI
    if avg_meas:
        a_ina = (a0_ina + a1_ina) / 2.0
        a_hat = (a0_inI + a1_inI) / 2.0
        w_inw = (w0_inw + w1_inw) / 2.0
        w_hat = (w0_inI + w1_inI) / 2.0
    return a_ina, a_hat, w_inw, w_hat


def whitener_rot_corrected(P_meas, d_th, Jl=None):
    """Sigma^-1/2 of an IMU factor's measurement covariance, corrected for the
    linearization point on the rotation block (shared by `ImuFactorACI_In`, 15x15,
    and the gyro-only factor, 6x6).

    Full congruence P <- T P_meas T^T with T = blkdiag(Jl(d_th), I): the column
    update reads the already-row-updated P, so the corner is Jl S Jl^T (symmetric,
    PD), not the one-sided S Jl^T the MVIS C++ uses, which is only valid near
    d_th = 0 and goes non-PD far from consistency (crashes the Cholesky). `Jl` may be
    passed in by a Jacobian path that needs it too. The inverse stays
    np.linalg.solve(L, I): solve_triangular would change the rounding.
    """
    if Jl is None:
        Jl = so3_left_jacobian(d_th)
    P = P_meas.copy()
    P[0:3, :] = Jl @ P_meas[0:3, :]
    P[:, 0:3] = P[:, 0:3] @ Jl.T
    P = 0.5 * (P + P.T)                       # symmetrize away float noise
    L = np.linalg.cholesky(P)
    return np.linalg.solve(L, np.eye(len(P_meas)))


def _state_recurrence(D_R, D_v, D_p, d_t, dR, int_R, iint_R, a_hat):
    """The ACI state recurrence of one step, verbatim from `feed_imu_single`:
    `D_p` uses the OLD `D_v`, `R_ki2k` is `D_R` before the update, and the
    products keep Python's grouping `(R_ki2k @ iint_R) @ a_hat`."""
    R_ki2k = D_R
    D_p = D_p + D_v * d_t + R_ki2k @ iint_R @ a_hat
    D_v = D_v + R_ki2k @ int_R @ a_hat
    D_R = D_R @ dR
    return D_R, D_v, D_p


def _phi_hwa(R_prev, dR, d_t, Jr, int_R, iint_R, int_RAJ, iint_RAJ, a_hat):
    """The ACI step's state transition phi_nn (9x9) and measurement Jacobian H_wa
    (9x6, over [w; a]), shared by `feed_imu_single` and `imu_segment._step`.
    `R_prev` is D_R before the step."""
    phi_nn = np.eye(9)
    phi_nn[0:3, 0:3] = dR.T
    phi_nn[3:6, 0:3] = -R_prev @ skew(iint_R @ a_hat)
    phi_nn[3:6, 6:9] = _I3 * d_t
    phi_nn[6:9, 0:3] = -R_prev @ skew(int_R @ a_hat)

    H_wa = np.zeros((9, 6))
    H_wa[0:3, 0:3] = Jr * d_t
    H_wa[3:6, 0:3] = -R_prev @ iint_RAJ
    H_wa[6:9, 0:3] = -R_prev @ int_RAJ
    H_wa[3:6, 3:6] = R_prev @ iint_R
    H_wa[6:9, 3:6] = R_prev @ int_R
    return phi_nn, H_wa


def step_input_jacobians(pack: _IntrPack, sign: int, a_ina, w_inw, w_hat, a_hat,
                         with_intrinsics: bool = True):
    """(H_b 6x6, H_in 6x24 | None) of one ACI step: the [w; a] input Jacobians
    over the biases [b_g, b_a] and over the 24-dim intrinsic tangent.
    `sign` = -1 is the reversed-time walk of `imu_segment._step`, +1 the forward
    step of `feed_imu_single`. Intrinsic tangent layout per model (see
    ImuModel): Kalibr/RPNG [Dw(6), Da(6), Tg(9), R(3)], Calyx [Dw(9), Da(6), Tg(9)]
    (no R block; R_wtoI = R_atoI = I)."""
    H_b = pack.Hb[sign]
    if not with_intrinsics:
        return H_b, None
    n, da_off, tg_off = pack.n_dw, pack.da_off, pack.tg_off
    R_atoI = pack.R_atoI
    daj = pack.daj(a_ina)                     # once per step, both rows
    H_in = np.zeros((6, 24))
    H_in[0:3, 0:n] = pack.sRw[sign] @ pack.dwj(w_inw)
    H_in[0:3, da_off:da_off + 6] = pack.nsMwTgRa[sign] @ daj
    H_in[3:6, da_off:da_off + 6] = R_atoI @ daj
    H_in[0:3, tg_off:tg_off + 9] = pack.nsMw[sign] @ pack.tgj(a_hat)
    if pack.has_R:
        r_off = pack.r_off
        if pack.model == KALIBR:
            H_in[0:3, r_off:r_off + 3] = -skew(w_hat) @ pack.R_wtoI
        else:
            H_in[0:3, r_off:r_off + 3] = pack.sMwTg[sign] @ skew(a_hat) @ R_atoI
            H_in[3:6, r_off:r_off + 3] = -skew(a_hat) @ R_atoI
    return H_b, H_in


def state_step(D_R, D_v, D_p, Dt, d_t, w_hat, a_hat):
    """One state-only ACI step (a speed-up): `integrate_R` plus the exact
    `feed_imu_single` recurrence, no Jacobians / covariance.

    Returns the new `(D_R, D_v, D_p, Dt)`; a step with `d_t <= 0` is skipped
    exactly as `feed_imu_single` skips it (inputs returned unchanged, `Dt` not
    advanced). The inputs are never mutated: every output is a new array, so a
    caller may keep the returned tuples as snapshots. `w_hat` / `a_hat` are
    the corrected measurements from `correct_measurements`.

    Bitwise to `feed_imu_single`'s `D_R / D_v / D_p / Dt` over the same sample
    sequence because `integrate_R` returns `integrate_RAJ(...)[:4]` bitwise
    and the recurrence is the same expression.
    """
    if d_t <= 0:
        return D_R, D_v, D_p, Dt
    dR, int_R, iint_R, _Jr = integrate_R(d_t, w_hat)
    D_R, D_v, D_p = _state_recurrence(D_R, D_v, D_p, d_t, dR, int_R, iint_R, a_hat)
    return D_R, D_v, D_p, Dt + d_t


class ACIPreintegrator:
    """Analytic combined IMU preintegrator with intrinsics (ports ACI_In).

    `track_intrinsics=False` skips the intrinsic (`J_*_in`) and gyro-accel
    time-offset (`J_*_tga`) Jacobian accumulation (a speed-up for callers
    that only read `D_*`, `J_*_bg/ba` and `P_meas`, e.g. the init's
    `linear_alignment`); those blocks stay at their `reset` zeros. `D_*`,
    the bias Jacobians and `P_meas` do not read them, so they are bitwise
    unchanged (asserted in `tests/cam_imu/test_densify_bitwise.py`).
    """

    def __init__(self, sigma_g, sigma_bg, sigma_a, sigma_ba,
                 model: int = RPNG, avg_meas: bool = False,
                 track_intrinsics: bool = True):
        self.model = model
        self.avg_meas = avg_meas
        self.track_intrinsics = bool(track_intrinsics)
        self.sigma_g, self.sigma_bg = sigma_g, sigma_bg
        self.sigma_a, self.sigma_ba = sigma_a, sigma_ba

        Qc = np.zeros((12, 12))
        Qc[0:3, 0:3] = sigma_g ** 2 * _I3
        Qc[3:6, 3:6] = sigma_a ** 2 * _I3
        Qc[6:9, 6:9] = sigma_bg ** 2 * _I3
        Qc[9:12, 9:12] = sigma_ba ** 2 * _I3
        self.Qc = Qc

        self.grav = np.zeros(3)
        self.intr = create_imu_model(model)
        self.b_g_init = np.zeros(3)
        self.b_a_init = np.zeros(3)
        self.reset(np.zeros(3), np.zeros(3))

    def reset(self, b_g_init: np.ndarray, b_a_init: np.ndarray):
        """Reset the preintegrator to identity at the given init biases (clears D_R/D_v/D_p, all bias + intrinsic Jacobians, and the covariance)."""
        self.b_g_init = np.asarray(b_g_init, float).copy()
        self.b_a_init = np.asarray(b_a_init, float).copy()
        self.Dt = 0.0
        self.D_R = np.eye(3)
        self.D_v = np.zeros(3)
        self.D_p = np.zeros(3)
        self.R_ki2k = np.eye(3)
        # bias Jacobians
        self.J_R_bg = np.zeros((3, 3)); self.J_R_ba = np.zeros((3, 3))
        self.J_v_bg = np.zeros((3, 3)); self.J_v_ba = np.zeros((3, 3))
        self.J_p_bg = np.zeros((3, 3)); self.J_p_ba = np.zeros((3, 3))
        # intrinsic Jacobians (3x24 each)
        self.J_R_in = np.zeros((3, 24))
        self.J_v_in = np.zeros((3, 24))
        self.J_p_in = np.zeros((3, 24))
        # gyro<->accel time-offset Jacobians (3,) — d(delta)/d t_ga at t_ga=0,
        # where the accel applying at gyro time t is a_m(t + t_ga)
        self.J_R_tga = np.zeros(3)
        self.J_v_tga = np.zeros(3)
        self.J_p_tga = np.zeros(3)
        # covariance (small prior, matches C++ 1e-15·I)
        self.P_meas = np.eye(15) * 1e-15

    def set_intrinsics(self, intr: ImuModel):
        """Set the IMU intrinsic model used to correct raw measurements during preintegration."""
        self.intr = intr

    def feed_imu_interval(self, t_vec, w_vec, a_vec):
        """Preintegrate a block of IMU samples pairwise; also caches the corrected w/a at the interval endpoints for the multi-IMU time-offset coupling."""
        t_vec = np.asarray(t_vec, float)
        w_vec = [np.asarray(w, float) for w in w_vec]
        a_vec = [np.asarray(a, float) for a in a_vec]
        pack = _IntrPack(self.intr)                   # once per interval
        for i in range(len(t_vec) - 1):
            self.feed_imu_single(t_vec[i], t_vec[i + 1],
                                 w_vec[i], a_vec[i], w_vec[i + 1], a_vec[i + 1],
                                 pack=pack)
        # corrected angular/linear measurements at interval endpoints (in IMU
        # frame) — used by the multi-IMU factor's kinematic time-offset coupling.
        # (Ma = R_atoI @ Da, Mw = R_wtoI @ Dw: the original left-to-right grouping.)
        Ma, Mw, Tg = pack.Ma, pack.Mw, pack.Tg
        self.a_m_s = Ma @ (a_vec[0] - self.b_a_init)
        self.a_m_e = Ma @ (a_vec[-1] - self.b_a_init)
        self.w_m_s = Mw @ (w_vec[0] - self.b_g_init - Tg @ self.a_m_s)
        self.w_m_e = Mw @ (w_vec[-1] - self.b_g_init - Tg @ self.a_m_e)

    def feed_imu_single(self, t_0, t_1, w_m_0, a_m_0, w_m_1=None, a_m_1=None,
                        pack: _IntrPack | None = None):
        """Preintegrate one IMU step over [t_0, t_1] (forward/endpoint measurements), updating D_R/D_v/D_p and their bias + intrinsic Jacobians.

        `pack` is the `_IntrPack` of `self.intr` (built once per interval by
        `feed_imu_interval`); None builds it for this step."""
        d_t = t_1 - t_0
        if d_t <= 0:
            return
        a1_given = a_m_1 is not None      # finite-diff a-dot needs a real endpoint
        if w_m_1 is None:
            w_m_1 = np.zeros(3)
        if a_m_1 is None:
            a_m_1 = np.zeros(3)

        if pack is None:
            pack = _IntrPack(self.intr)
        Ma = pack.Ma
        track_in = self.track_intrinsics

        # corrected measurements (intrinsics + bias), both endpoints + average
        # (the helper shared with the state-only walk; same expressions)
        a_ina, a_hat, w_inw, w_hat = correct_measurements(
            pack, self.b_g_init, self.b_a_init, w_m_0, a_m_0, w_m_1, a_m_1,
            self.avg_meas)

        dR, int_R, iint_R, Jr, int_RAJ, iint_RAJ = integrate_RAJ(d_t, w_hat, a_hat)

        self.R_ki2k = self.D_R.copy()
        # state recurrence (D_p with the OLD D_v; shared with `state_step`)
        self.D_R, self.D_v, self.D_p = _state_recurrence(
            self.D_R, self.D_v, self.D_p, d_t, dR, int_R, iint_R, a_hat)

        # --- Jacobian blocks (prefix products from the pack, see _IntrPack) ---
        H_b, H_in = step_input_jacobians(pack, 1, a_ina, w_inw, w_hat, a_hat, track_in)
        H_n = H_b

        phi_nn, H_wa = _phi_hwa(self.R_ki2k, dR, d_t, Jr, int_R, iint_R,
                                int_RAJ, iint_RAJ, a_hat)

        # accumulate bias Jacobians (9x6 stacked: theta,p,v over bg,ba)
        J_b_k = np.zeros((9, 6))
        J_b_k[0:3, 0:3] = self.J_R_bg; J_b_k[0:3, 3:6] = self.J_R_ba
        J_b_k[3:6, 0:3] = self.J_p_bg; J_b_k[3:6, 3:6] = self.J_p_ba
        J_b_k[6:9, 0:3] = self.J_v_bg; J_b_k[6:9, 3:6] = self.J_v_ba
        J_b_ki = phi_nn @ J_b_k + H_wa @ H_b
        self.J_R_bg, self.J_R_ba = J_b_ki[0:3, 0:3], J_b_ki[0:3, 3:6]
        self.J_p_bg, self.J_p_ba = J_b_ki[3:6, 0:3], J_b_ki[3:6, 3:6]
        self.J_v_bg, self.J_v_ba = J_b_ki[6:9, 0:3], J_b_ki[6:9, 3:6]

        if track_in:
            # accumulate intrinsic Jacobians (9x24 stacked)
            J_in_k = np.zeros((9, 24))
            J_in_k[0:3, :] = self.J_R_in
            J_in_k[3:6, :] = self.J_p_in
            J_in_k[6:9, :] = self.J_v_in
            J_in_ki = phi_nn @ J_in_k + H_wa @ H_in
            self.J_R_in = J_in_ki[0:3, :]
            self.J_p_in = J_in_ki[3:6, :]
            self.J_v_in = J_in_ki[6:9, :]

            # accumulate gyro<->accel time-offset Jacobian (one extra column of
            # the intrinsic pattern): da/dt_ga = a-dot (raw-measurement finite
            # difference over the step), entering the accel row directly and
            # the gyro row through the g-sensitivity Tg.
            a_dot = (a_m_1 - a_m_0) / d_t if a1_given else np.zeros(3)
            da_inI_tga = Ma @ a_dot
            H_tga = np.concatenate([pack.nMwTg @ da_inI_tga, da_inI_tga])
            J_tga_k = np.concatenate([self.J_R_tga, self.J_p_tga, self.J_v_tga])
            J_tga_ki = phi_nn @ J_tga_k + H_wa @ H_tga
            self.J_R_tga = J_tga_ki[0:3]
            self.J_p_tga = J_tga_ki[3:6]
            self.J_v_tga = J_tga_ki[6:9]

        # covariance propagation
        Phi = np.eye(15)
        Phi[0:9, 0:9] = phi_nn
        Phi[0:9, 9:15] = H_wa @ H_b
        G = np.zeros((15, 12))
        G[0:9, 0:6] = H_wa @ H_n
        G[9:12, 6:9] = _I3 * d_t
        G[12:15, 9:12] = _I3 * d_t
        Qd = self.Qc / d_t
        self.P_meas = Phi @ self.P_meas @ Phi.T + G @ Qd @ G.T

        self.R_ki2k = self.D_R.copy()
        self.Dt += d_t


class ACIGyroPreintegrator:
    """Gyroscope-only ACI preintegration (rotation only), ports the C++ MVIS `ACI_gyro`.

    Integrates raw gyro (w_m) into D_R plus the gyro-bias Jacobian (3x3) and the
    gyro-intrinsic (Dw, 3x6) Jacobian and the 6x6 [theta, b_g] covariance — the
    quantities the gyro-only factor `ImuFactorACI_Multi_In_SO3` consumes. Only
    `Dw` matters (no accel/Tg); R_wtoI is held at identity in the recurrence, as
    in the C++.
    """

    def __init__(self, sigma_g, sigma_bg, model: int = RPNG, avg_meas: bool = False):
        self.model = model
        self.avg_meas = avg_meas
        self.sigma_g, self.sigma_bg = sigma_g, sigma_bg
        Qc = np.zeros((6, 6))
        Qc[0:3, 0:3] = sigma_g ** 2 * _I3
        Qc[3:6, 3:6] = sigma_bg ** 2 * _I3
        self.Qc = Qc
        self.intr = create_imu_model(model)   # only x_dw / Dw() used
        self.reset(np.zeros(3))

    def reset(self, b_g_init):
        """Reset the preintegrator to identity at the given init biases (clears D_R/D_v/D_p, all bias + intrinsic Jacobians, and the covariance)."""
        self.b_g_init = np.asarray(b_g_init, float).copy()
        self.Dt = 0.0
        self.D_R = np.eye(3)
        self.R_ki2k = np.eye(3)
        self.J_R_bg = np.zeros((3, 3))
        self.J_R_in = np.zeros((3, n_dw_for(self.model)))   # w.r.t. x_Dw (6 or 9)
        self.P_meas = np.eye(6) * 1e-15

    def set_intrinsics(self, intr: ImuModel):
        """Set the IMU intrinsic model used to correct raw measurements during preintegration."""
        self.intr = intr

    def feed_imu_interval(self, t_vec, w_vec):
        """Preintegrate a block of IMU samples pairwise; also caches the corrected w/a at the interval endpoints for the multi-IMU time-offset coupling."""
        t_vec = np.asarray(t_vec, float)
        w_vec = [np.asarray(w, float) for w in w_vec]
        for i in range(len(t_vec) - 1):
            self.feed_imu_single(t_vec[i], t_vec[i + 1], w_vec[i], w_vec[i + 1])
        Dw = self.intr.Dw()                       # R_wtoI ≡ I
        self.w_m_s = Dw @ (w_vec[0] - self.b_g_init)
        self.w_m_e = Dw @ (w_vec[-1] - self.b_g_init)

    def feed_imu_single(self, t_0, t_1, w_m_0, w_m_1=None):
        """Preintegrate one IMU step over [t_0, t_1] (forward/endpoint measurements), updating D_R/D_v/D_p and their bias + intrinsic Jacobians."""
        d_t = t_1 - t_0
        if d_t <= 0:
            return
        if w_m_1 is None:
            w_m_1 = np.zeros(3)
        Dw = self.intr.Dw()                       # R_wtoI ≡ I in the gyro recurrence
        w0_inw = w_m_0 - self.b_g_init
        w1_inw = w_m_1 - self.b_g_init
        w0_hat = Dw @ w0_inw
        w1_hat = Dw @ w1_inw
        w_inw, w_hat = w0_inw, w0_hat
        if self.avg_meas:
            w_inw = (w0_inw + w1_inw) / 2.0
            w_hat = (w0_hat + w1_hat) / 2.0

        dR, _int_R, _iint_R, Jr, _, _ = integrate_RAJ(d_t, w_hat, np.zeros(3))
        self.R_ki2k = self.D_R.copy()
        self.D_R = self.D_R @ dR

        # C++ ACI_gyro writes `- Jr*dt` here, dropping the Dw factor (which it
        # *does* keep in Phi/G below) — a latent approximation. FD requires the
        # Dw factor (R_wtoI ≡ I), matching the full ACI_In bias-Jacobian form.
        self.J_R_bg = dR.T @ self.J_R_bg - (Jr * d_t) @ Dw
        self.J_R_in = dR.T @ self.J_R_in + (Jr * d_t) @ dw_jacobian(w_inw, self.model)

        Phi = np.eye(6)
        Phi[0:3, 0:3] = dR.T
        Phi[0:3, 3:6] = -(Jr * d_t) @ Dw
        G = np.zeros((6, 6))
        G[0:3, 0:3] = -(Jr * d_t) @ Dw
        G[3:6, 3:6] = _I3 * d_t
        Qd = self.Qc / d_t
        self.P_meas = Phi @ self.P_meas @ Phi.T + G @ Qd @ G.T
        self.Dt += d_t
