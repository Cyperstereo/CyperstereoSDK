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
"""Upsample a slow IMU before integration.

At 100 Hz a 33-67 ms factor window holds very few samples, which biases that IMU's lever arm
(twin of mvis_4imus_3cams: GX3-35). Real samples are kept and new ones are inserted between them.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import CubicSpline

logger = logging.getLogger(__name__)

RATE_TOLERANCE = 0.95       # a nominal 200 Hz IMU that reads 199 Hz is not slow
MAX_GAP_FACTOR = 3.0        # do not invent samples across a dropout


def upsample_imu(t, w, a, min_hz: float = 200.0, method: str = "linear", name: str = ""):
    """(t, w, a) with at least min_hz between real samples; the same arrays if already fast enough."""
    t = np.asarray(t, float)
    w, a = np.asarray(w, float), np.asarray(a, float)
    if min_hz <= 0.0 or len(t) < 2:
        return t, w, a
    if method not in ("linear", "cubic"):
        raise ValueError(f"imu_upsample_method must be 'linear' or 'cubic', got {method!r}")
    dt = np.diff(t)
    med = float(np.median(dt))
    hz = 1.0 / med
    if hz >= min_hz * RATE_TOLERANCE:
        return t, w, a
    parts = int(np.ceil(min_hz / hz - 1e-9))
    inside = dt <= MAX_GAP_FACTOR * med
    new = [t[i] + dt[i] * np.arange(1, parts) / parts for i in np.nonzero(inside)[0]]
    t_out = np.sort(np.concatenate([t, *new]))
    if method == "linear":
        w_out = np.column_stack([np.interp(t_out, t, w[:, k]) for k in range(3)])
        a_out = np.column_stack([np.interp(t_out, t, a[:, k]) for k in range(3)])
    else:
        w_out, a_out = CubicSpline(t, w)(t_out), CubicSpline(t, a)(t_out)
    logger.info("IMU %s: %.1f Hz -> %.1f Hz (%s), %d -> %d samples, %d gaps left alone",
                name, hz, 1.0 / float(np.median(np.diff(t_out))), method, len(t), len(t_out),
                int(np.sum(~inside)))
    return t_out, w_out, a_out
