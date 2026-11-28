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
"""Motion-based calibration-segment selection: pick the most-exciting contiguous
window of a multi-IMU bag, scored on combined gyro+accel excitation."""
import numpy as np


def select_motion_window(imu_streams, segment_length=80.0, step=5.0):
    """Return absolute header-second (lo, hi) of the most-exciting contiguous
    `segment_length` window, or None if the bag spans < segment_length (an
    exactly-segment_length bag is selectable).

    Score per candidate window, summed over IMUs:
        rot   = mean(|w|²)                   (rotation excitation power)
        trans = mean((|a| - mean_win|a|)²)   (translation; gravity removed via window-mean)
    each divided by that IMU's whole-bag rot/trans power so gyro and accel and
    different IMUs weigh comparably. Slide in `step` increments; argmax (earliest
    start on ties)."""
    streams = [s for s in imu_streams
               if s.get("t") is not None and len(np.asarray(s["t"])) > 2]
    if not streams:
        return None
    t_lo = min(float(np.min(s["t"])) for s in streams)
    t_hi = max(float(np.max(s["t"])) for s in streams)
    if (t_hi - t_lo) < segment_length:
        return None
    prep = []
    for s in streams:
        t = np.asarray(s["t"], float)
        # nan_to_num: a dropped/corrupt IMU sample (NaN) must not poison the
        # whole score (NaN > best is always False -> silent earliest-start).
        wm = np.nan_to_num(np.linalg.norm(np.asarray(s["w"], float), axis=1))
        am = np.nan_to_num(np.linalg.norm(np.asarray(s["a"], float), axis=1))
        # whole-bag normalisation constants (power = mean of squared values)
        rsc = float(np.mean(wm ** 2)) or 1.0
        tsc = float(np.mean((am - am.mean()) ** 2)) or 1.0
        prep.append((t, wm, am, rsc, tsc))
    starts = np.arange(t_lo, t_hi - segment_length + 1e-9, step)
    best_lo, best_score = float(starts[0]), -1.0
    for s0 in starts:
        s1 = s0 + segment_length
        score = 0.0
        for (t, wm, am, rsc, tsc) in prep:
            m = (t >= s0) & (t < s1)
            if int(m.sum()) < 2:
                continue
            score += float(np.mean(wm[m] ** 2)) / rsc \
                + float(np.mean((am[m] - am[m].mean()) ** 2)) / tsc
        if score > best_score:
            best_score, best_lo = score, float(s0)
    return best_lo, best_lo + segment_length
