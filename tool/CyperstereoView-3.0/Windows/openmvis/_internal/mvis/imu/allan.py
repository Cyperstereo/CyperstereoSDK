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
"""Allan deviation and static-span gravity from raw IMU streams (no solving).

Library helpers for characterising an IMU before calibration: `read_imu` loads a
EuRoC `imu0/data.csv`, `allan_dev` gives the overlapping Allan deviation of one axis
at the cluster times you pass, and `static_gravity` measures |g| over the static
seconds of a sequence. Use them to sanity-check the `imu.sigma_*` noise values of a
config against a static recording (see doc/estimate_config.md).
"""
from pathlib import Path

import numpy as np

GRAV = 9.80665


def read_imu(path):
    """EuRoC imu csv (`t_ns, w xyz, a xyz`, one header line) -> t (s), w, a."""
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    return d[:, 0] / 1e9, d[:, 1:4], d[:, 4:7]


def allan_dev(x, fs, taus):
    """Overlapping Allan deviation of rate signal x at cluster times taus."""
    theta = np.cumsum(x, dtype=np.float64) / fs
    n = len(theta)
    out = []
    for tau in taus:
        m = int(round(tau * fs))
        if m < 1 or 2 * m >= n:
            out.append(np.nan)
            continue
        d = theta[2 * m:] - 2 * theta[m:-m] + theta[:-2 * m]
        out.append(np.sqrt(0.5 * np.mean(d ** 2)) / tau)
    return np.array(out)


def static_gravity(seq_imu_dir):
    """Median |a| (csv + raw) over detected static 1 s windows; None if <5 s static."""
    out = {}
    for name in ("data.csv", "data.raw.csv"):
        p = Path(seq_imu_dir) / name
        if not p.exists():
            out[name] = None
            continue
        t, w, a = read_imu(p)
        fs = max((len(t) - 1) / (t[-1] - t[0]), 1.0)
        win = max(int(fs), 10)
        nwin = len(t) // win
        wr = w[:nwin * win].reshape(nwin, win, 3)
        ar = a[:nwin * win].reshape(nwin, win, 3)
        spread = wr.std(axis=1).max(axis=1)   # per-window worst-axis gyro std
        # raw streams are LSB-quantized: the at-rest floor is ~0.015 rad/s
        # (measured on MOC imustatic), so gate at 0.03 — motion is >>0.1
        static = spread < 0.03
        if static.sum() < 5:
            out[name] = None
            continue
        mags = np.linalg.norm(ar[static].reshape(-1, 3), axis=1)
        out[name] = {"static_s": int(static.sum()),
                     "grav_median": float(np.median(mags)),
                     "pct": round((float(np.median(mags)) / GRAV - 1) * 100, 2)}
    return out
