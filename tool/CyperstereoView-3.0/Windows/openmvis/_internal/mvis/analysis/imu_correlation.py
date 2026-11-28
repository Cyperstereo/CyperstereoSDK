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
"""Estimator-free gyro->accel time offset (t_ga) from raw IMU correlation.

Physics (tier 1, kinematic identity): with the accelerometer dominated by gravity in the body
frame, da/dt = -omega x a (+ translation jerk). The gyro therefore predicts the accel derivative
sample by sample; the lag that maximises the correlation between the measured da/dt and the
gyro-based prediction is a solver-free measurement of the gyro<->accel clock offset
(`tier1_curve`). `tier1_lever_curve` also fits the rotation-centre lever r per lag (the handheld
rotation-translation coupling a_trans = wd x r + w x (w x r)) so it no longer biases the lag.
Tier 2 (`tier2_curve`, the envelope correlation of |omega| and |a|) is a coarse sanity check.

Sign convention: the curves shift the gyro by +lag. The peak lag* means the accel stream lags the
gyro by lag*, so MVIS's ACI t_ga (gyro clock is the reference; the accel sample stamped t was
taken at t - t_ga) is -lag*. A synthetic control locks this: re-stamping the accel by +s moves
the reported t_ga by +s.

Use: with raw gyro `(tg, w)` and accel `(ta, a)` arrays (`mvis.io.cam_imu_loader.read_imu_only`
gives them per IMU), call `tier1_curve(tg, w, ta, a, lags, band, fs)` over a grid of lags
and a pass band, and `peak` on the curve; the sign rule above turns the peak lag into a
solver-free check of the `t_ga` an MVIS cam-IMU solve reports.
"""
import numpy as np
from scipy import signal


def bandpass(x, fs, lo, hi, axis=0):
    """Zero-phase 2nd-order Butterworth bandpass; the upper edge is capped at 0.45 fs."""
    hi = min(hi, 0.45 * fs)
    sos = signal.butter(2, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=axis)


def tier1_curve(tg, w, ta, a, lags, band, fs):
    """corr(da/dt(t), -(omega(t+lag) x a(t))) summed over axes, per lag."""
    # uniform accel grid
    tt = np.arange(ta[0], ta[-1], 1.0 / fs)
    a_u = np.column_stack([np.interp(tt, ta, a[:, k]) for k in range(3)])
    a_f = bandpass(a_u, fs, band[0], band[1])
    dadt = np.gradient(a_f, 1.0 / fs, axis=0)
    # rotation-dominance weights (translation-jerk robustness)
    w_u0 = np.column_stack([np.interp(tt, tg, w[:, k]) for k in range(3)])
    pred0 = -np.cross(w_u0, a_u)
    mag = np.linalg.norm(pred0, axis=1)
    wgt = (mag > np.median(mag)).astype(float)
    out = []
    for lag in lags:
        w_u = np.column_stack([np.interp(tt + lag, tg, w[:, k]) for k in range(3)])
        pred = -np.cross(w_u, a_u)
        pred = bandpass(pred, fs, band[0], band[1])
        num = np.sum(wgt[:, None] * dadt * pred)
        den = np.sqrt(np.sum(wgt[:, None] * dadt**2) * np.sum(wgt[:, None] * pred**2))
        out.append(num / den if den > 0 else 0.0)
    return np.array(out)


def tier1_lever_curve(tg, w, ta, a, lags, band, fs):
    """Tier 1 with the lever-term nuisance, unit weights: per lag, fit r (3 params, linear least
    squares) in da/dt ~ bp(-omega x a) + d/dt bp(wd x r + omega x (omega x r)), with bp the
    `bandpass`, then correlate da/dt with that model. Returns `(curve, r_fits)`, r in m per lag."""
    tt = np.arange(ta[0], ta[-1], 1.0 / fs)
    a_u = np.column_stack([np.interp(tt, ta, a[:, k]) for k in range(3)])
    a_f = bandpass(a_u, fs, band[0], band[1])
    dadt = np.gradient(a_f, 1.0 / fs, axis=0)
    curve, r_fits = [], []
    for lag in lags:
        w_u = np.column_stack([np.interp(tt + lag, tg, w[:, k]) for k in range(3)])
        wd = np.gradient(w_u, 1.0 / fs, axis=0)
        base = -np.cross(w_u, a_u)                      # gravity-rotation term
        # lever regressors: the IMU's translation accel wd x r + w x (w x r) is linear in r; its
        # contribution to da/dt is the time derivative, one column per axis e of r
        cols = []
        for e in np.eye(3):
            E = e[None, :] * np.ones_like(w_u)
            L = bandpass(np.cross(wd, E) + np.cross(w_u, np.cross(w_u, E)), fs, band[0], band[1])
            cols.append(np.gradient(L, 1.0 / fs, axis=0))
        base_f = bandpass(base, fs, band[0], band[1])
        y = (dadt - base_f).reshape(-1)
        A = np.column_stack([c.reshape(-1) for c in cols])
        r_fit, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        model = base_f + sum(r_fit[j] * cols[j] for j in range(3))
        num = np.sum(dadt * model)
        den = np.sqrt(np.sum(dadt**2) * np.sum(model**2))
        curve.append(num / den)
        r_fits.append(r_fit.copy())
    return np.array(curve), np.array(r_fits)


def tier2_curve(tg, w, ta, a, lags, band, fs):
    """Coarse check: corr of the |omega| and |a| band envelopes, per lag."""
    tt = np.arange(max(tg[0], ta[0]), min(tg[-1], ta[-1]), 1.0 / fs)
    wn = np.interp(tt, tg, np.linalg.norm(w, axis=1))
    an = np.interp(tt, ta, np.linalg.norm(a, axis=1))
    wn = bandpass(wn, fs, band[0], band[1])
    an = bandpass(an, fs, band[0], band[1])
    e_w = bandpass(np.abs(wn), fs, 0.05, 5.0)
    out = []
    for lag in lags:
        e_ws = np.interp(tt + lag, tt, e_w)
        e_a = bandpass(np.abs(an), fs, 0.05, 5.0)
        c = np.corrcoef(e_ws, e_a)[0, 1]
        out.append(c)
    return np.array(out)


def peak(lags, curve):
    """(lag, value) at the curve maximum, the lag refined by a parabola when interior."""
    i = int(np.argmax(curve))
    if 0 < i < len(lags) - 1:  # parabolic refinement
        y0, y1, y2 = curve[i - 1], curve[i], curve[i + 1]
        d = (y0 - y2) / (2 * (y0 - 2 * y1 + y2)) if (y0 - 2 * y1 + y2) != 0 else 0.0
        return lags[i] + d * (lags[1] - lags[0]), curve[i]
    return lags[i], curve[i]
