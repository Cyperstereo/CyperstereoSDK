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
"""Synthetic IMU measurements from a BsplineSE3 trajectory.

Inverts the ACI measurement model (see `factors/aci.py`) so that an
`ACIPreintegrator` fed the *true* `ImuIntrinsics` and the *true* initial biases
reconstructs the trajectory's preintegration exactly in the noise-free case.

ACI correction (measurement -> true), with biases b_g/b_a:

    a_ina = a_m - b_a
    a_inI = R_atoI · Da · a_ina
    w_inw = w_m - b_g - Tg · a_inI
    w_inI = R_wtoI · Dw · w_inw

Generation is the exact inverse, given the trajectory's true body-frame angular
velocity w_inI = w_IinI and specific force a_inI = R_GtoI (a_IinG - g_inG):

    a_m = Da^{-1} R_atoI^T a_inI + b_a            (+ white noise)
    w_m = Dw^{-1} R_wtoI^T w_inI + Tg a_inI + b_g (+ white noise)

Biases follow a discrete random walk; white-noise and bias-walk sigmas are
continuous-time densities (discretized by the sample dt). All sigmas default to 0
(noise-free), which is what the optimization-recovery test uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mvis.calibration.factors.lie_groups import skew

GRAVITY = np.array([0.0, 0.0, -9.81])


def true_imu(traj, t: float, g_inG=GRAVITY):
    """Body-frame true angular velocity and specific force at time t.

    Returns (w_inI, a_inI): w_inI = w_IinI, a_inI = R_GtoI (a_IinG - g_inG).
    """
    R_GtoI, _p, w_IinI, _v, _alpha, a_IinG = traj.get_acceleration(t)
    a_inI = R_GtoI @ (a_IinG - np.asarray(g_inG, float))
    return w_IinI, a_inI


def aux_imu_true(traj, t: float, R_IstoI, p_IsinI, g_inG=GRAVITY):
    """Body-frame true angular velocity and specific force of an AUXILIARY IMU.

    The aux IMU is rigidly mounted at extrinsic (R_IstoI: aux->base rotation,
    p_IsinI: aux origin in the base frame). It shares the base angular velocity
    (rotated into its frame) and feels the lever-arm acceleration. Matches MVIS
    `SimulatorCalib::get_next_imu_s` (SimulatorCalib.cpp:620-624):

        w_inIs  = R_IstoI^T · w_IinI
        a_IsinG = a_IinG + R_ItoG ([w]x^2 + [alpha]x) p_IsinI      (lever arm)
        a_inIs  = R_IstoG^T (a_IsinG - g_inG)                      (specific force)

    Reduces exactly to `true_imu` for identity extrinsic (R=I, p=0).
    """
    R_GtoI, _p, w_IinI, _v, alpha_IinI, a_IinG = traj.get_acceleration(t)
    R_ItoG = R_GtoI.T
    R_IstoI = np.asarray(R_IstoI, float)
    p_IsinI = np.asarray(p_IsinI, float)
    sk_w = skew(w_IinI)
    a_IsinG = a_IinG + R_ItoG @ (sk_w @ sk_w + skew(alpha_IinI)) @ p_IsinI
    R_IstoG = R_ItoG @ R_IstoI
    a_inIs = R_IstoG.T @ (a_IsinG - np.asarray(g_inG, float))
    w_inIs = R_IstoI.T @ w_IinI
    return w_inIs, a_inIs


def measure_imu(w_inI, a_inI, intr, b_g, b_a):
    """Invert the ACI correction: true (w_inI, a_inI) -> raw (w_m, a_m).

    Thin wrapper over `ImuModel.measure` (the model owns the forward/inverse
    measurement math); kept as a free function for `simulate_imu`'s loop.
    """
    return intr.measure(w_inI, a_inI, b_g, b_a)


def imu_times(t_start: float, t_end: float, rate: float) -> np.ndarray:
    """Uniform sample times spanning [t_start, t_end] at ~`rate` Hz, with both
    endpoints hit exactly (robust to floating-point in the span)."""
    n = int(round((t_end - t_start) * rate)) + 1
    return np.linspace(t_start, t_end, max(n, 2))


@dataclass
class ImuStream:
    """A sampled IMU stream plus the ground truth used to generate it."""
    t: np.ndarray        # (N,)
    w_m: np.ndarray      # (N,3) raw gyro
    a_m: np.ndarray      # (N,3) raw accel
    b_g: np.ndarray      # (N,3) ground-truth gyro bias per sample
    b_a: np.ndarray      # (N,3) ground-truth accel bias per sample
    w_true: np.ndarray   # (N,3) true w_inI
    a_true: np.ndarray   # (N,3) true a_inI (specific force)


def simulate_imu(traj, times, intr, g_inG=GRAVITY, R_IstoI=None, p_IsinI=None,
                 td=0.0, b_g0=None, b_a0=None,
                 sigma_w=0.0, sigma_a=0.0, sigma_bg=0.0, sigma_ba=0.0,
                 seed=0) -> ImuStream:
    """Generate an `ImuStream` by sampling `traj` at `times`.

    Base IMU: leave R_IstoI/p_IsinI/td at their defaults (identity, no offset) —
    the trajectory frame *is* the IMU. Auxiliary IMU: pass its extrinsic
    (R_IstoI aux->base, p_IsinI aux origin in base frame) and time offset td;
    the lever-arm acceleration is added and the trajectory is sampled at `t + td`
    (td > 0 ⇒ the aux clock lags the base). `intr` is then the aux intrinsics.

    sigma_w / sigma_a are continuous-time white-noise densities (added as
    sigma/sqrt(dt) per sample); sigma_bg / sigma_ba drive the bias random walk
    (sigma*sqrt(dt) per step). Defaults are noise-free.
    """
    times = np.asarray(times, float)
    rng = np.random.RandomState(seed)
    g_inG = np.asarray(g_inG, float)
    b_g = np.zeros(3) if b_g0 is None else np.array(b_g0, float)
    b_a = np.zeros(3) if b_a0 is None else np.array(b_a0, float)
    td = float(td)
    is_aux = R_IstoI is not None or p_IsinI is not None or td != 0.0
    R_ext = np.eye(3) if R_IstoI is None else np.asarray(R_IstoI, float)
    p_ext = np.zeros(3) if p_IsinI is None else np.asarray(p_IsinI, float)

    n = len(times)
    w_m = np.zeros((n, 3)); a_m = np.zeros((n, 3))
    bg = np.zeros((n, 3)); ba = np.zeros((n, 3))
    wt = np.zeros((n, 3)); at = np.zeros((n, 3))

    for k, t in enumerate(times):
        t_eval = t + td
        if is_aux:
            w_inI, a_inI = aux_imu_true(traj, t_eval, R_ext, p_ext, g_inG)
        else:
            w_inI, a_inI = true_imu(traj, t_eval, g_inG)
        mw, ma = measure_imu(w_inI, a_inI, intr, b_g, b_a)
        dt = (times[k + 1] - times[k]) if k + 1 < n else (times[k] - times[k - 1])
        if sigma_w:
            mw = mw + sigma_w / np.sqrt(dt) * rng.standard_normal(3)
        if sigma_a:
            ma = ma + sigma_a / np.sqrt(dt) * rng.standard_normal(3)
        w_m[k], a_m[k] = mw, ma
        bg[k], ba[k] = b_g, b_a
        wt[k], at[k] = w_inI, a_inI
        if sigma_bg:
            b_g = b_g + sigma_bg * np.sqrt(dt) * rng.standard_normal(3)
        if sigma_ba:
            b_a = b_a + sigma_ba * np.sqrt(dt) * rng.standard_normal(3)

    return ImuStream(times, w_m, a_m, bg, ba, wt, at)
