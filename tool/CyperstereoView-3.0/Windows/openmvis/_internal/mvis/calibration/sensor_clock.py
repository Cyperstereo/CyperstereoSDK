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
"""Sensor-clock recovery for cameras whose stamps do not follow their sensor.

Some USB cameras free-run on their own sensor clock while the driver publishes
and stamps frames on a different, regular cadence (e.g. the ELP stereo under
usb_cam 0.3.6: sensor ~29.62 Hz, frames stamped by a 25 Hz polling loop after
dequeue + decode). Each published frame is then off its stamp by a varying
amount — a sawtooth spanning one sensor period — which a single td cannot
absorb and which biases the joint calibration through errors-in-variables.

Given per-frame capture offsets delta_f (e.g. solved with
`solve.vision_imu_frame_dt_sigma`), the true capture times c_f = t_f + delta_f
lie on a lattice phi + n_f * T_s. This module finds that lattice (period T_s,
phase phi, integer sensor index n_f per frame) and checks it, so the frames can
be retimed onto the sensor clock with two parameters instead of one free
offset per frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SensorClock:
    period: float                 # T_s (s)
    phase: float                  # phi (s): capture time of sensor index 0
    index: np.ndarray             # n_f per input frame (int)
    residual: np.ndarray          # c_f - (phi + n_f T_s) per frame (s)
    coherence: float              # |mean exp(2 pi i c/T_s)| in [0, 1]; 1 = on the lattice
    scatter: float                # robust std of the residual (s)
    inlier: np.ndarray = field(default=None)   # frames used in the final fit (bool)

    def capture_times(self) -> np.ndarray:
        """Lattice capture times phi + n_f T_s for every input frame."""
        return self.phase + self.index * self.period

    def report(self) -> dict:
        return {"period_s": float(self.period), "rate_hz": float(1.0 / self.period),
                "phase_s": float(self.phase), "coherence": float(self.coherence),
                "scatter_ms": float(self.scatter * 1e3),
                "n_frames": int(len(self.index)), "n_inliers": int(np.sum(self.inlier))}


def _coherence(c: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Lattice coherence of times c for each candidate period (vectorised)."""
    c = c - c[0]
    out = np.empty(len(periods))
    for i0 in range(0, len(periods), 512):
        p = periods[i0:i0 + 512]
        ph = np.exp(2j * np.pi * np.outer(1.0 / p, c))
        out[i0:i0 + 512] = np.abs(ph.mean(axis=1))
    return out


def fit_sensor_clock(capture_times, period_range=(1.0 / 120.0, 1.0 / 5.0),
                     period_hint: float | None = None, grid_step: float = 2e-6,
                     harmonic_tol: float = 0.85, max_iter: int = 5) -> SensorClock:
    """Find the sensor lattice phi + n T_s that the capture times sit on.

    The coherence |mean exp(2 pi i c / T)| is high at the true period AND at its
    integer fractions (T/2, T/3 ... are lattices too), so among the candidates
    whose coherence reaches `harmonic_tol` x the global maximum the LONGEST
    period is taken. `period_hint` narrows the search to +-20 % around it.
    The phase and period are then refined by least squares on the integer
    indices, rejecting > 4 MAD outliers. Raises ValueError for < 10 frames."""
    c = np.sort(np.asarray(capture_times, float))
    order = np.argsort(np.asarray(capture_times, float))
    if len(c) < 10:
        raise ValueError("fit_sensor_clock needs at least 10 frames")
    lo, hi = period_range
    if period_hint is not None:
        lo, hi = max(lo, 0.8 * period_hint), min(hi, 1.2 * period_hint)
    # coarse grid (step sized so the phase walks < 0.05 cycle over the span), then fine
    span = c[-1] - c[0]
    coarse = max(grid_step, 0.05 * lo * lo / max(span, 1e-9))
    grid = np.arange(lo, hi, coarse)
    R = _coherence(c, grid)
    cand = grid[R >= harmonic_tol * R.max()]
    # group adjacent candidates into peaks, keep the longest-period peak
    peaks, cur = [], [cand[0]]
    for p in cand[1:]:
        if p - cur[-1] <= 1.5 * coarse:
            cur.append(p)
        else:
            peaks.append(cur); cur = [p]
    peaks.append(cur)
    best_peak = peaks[-1]
    fine = np.arange(best_peak[0] - 2 * coarse, best_peak[-1] + 2 * coarse, grid_step)
    Rf = _coherence(c, fine)
    T = float(fine[int(np.argmax(Rf))])

    ang = np.angle(np.mean(np.exp(2j * np.pi * (c - c[0]) / T)))
    phi = c[0] + (ang % (2 * np.pi)) / (2 * np.pi) * T
    inl = np.ones(len(c), bool)
    for _ in range(max_iter):
        n = np.round((c - phi) / T).astype(int)
        A = np.column_stack([np.ones(int(inl.sum())), n[inl]])
        (phi, T), *_ = np.linalg.lstsq(A, c[inl], rcond=None)
        res = c - (phi + n * T)
        mad = 1.4826 * np.median(np.abs(res[inl] - np.median(res[inl])))
        new = np.abs(res - np.median(res[inl])) <= max(4.0 * mad, 1e-6)
        if np.array_equal(new, inl):
            break
        inl = new
    n = np.round((c - phi) / T).astype(int)
    res = c - (phi + n * T)
    coh = float(_coherence(c, np.array([T]))[0])
    scatter = float(1.4826 * np.median(np.abs(res[inl] - np.median(res[inl]))))
    # back to input order
    inv = np.empty_like(order); inv[order] = np.arange(len(order))
    return SensorClock(period=float(T), phase=float(phi), index=n[inv], residual=res[inv],
                       coherence=coh, scatter=scatter, inlier=inl[inv])


def check_sensor_clock(clock: SensorClock, stamps, min_coherence: float = 0.6,
                       max_scatter_frac: float = 1.0 / 6.0) -> list:
    """Reasons the lattice should NOT be used (empty list = usable).

    - coherence below `min_coherence` (capture times are not on one clock);
    - scatter above `max_scatter_frac` of the period (indices ambiguous);
    - two inlier frames on the same sensor index, or indices out of stamp order.
    Outliers (`clock.inlier` False) are not judged: they are not retimed."""
    reasons = []
    if clock.coherence < min_coherence:
        reasons.append(f"lattice coherence {clock.coherence:.2f} < {min_coherence}")
    if clock.scatter > max_scatter_frac * clock.period:
        reasons.append(f"scatter {clock.scatter*1e3:.2f} ms > {max_scatter_frac:.2f} period")
    stamps = np.asarray(stamps, float)
    inl = clock.inlier if clock.inlier is not None else np.ones(len(stamps), bool)
    order = np.argsort(stamps[inl])
    steps = np.diff(clock.index[inl][order])
    if np.any(steps <= 0):
        reasons.append(f"{int(np.sum(steps <= 0))} frames share or reverse a sensor index")
    return reasons
