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
"""Shared corner bookkeeping of the three cam-IMU reprojection factors.

`CameraCalibInterpFactor`, `CameraCalibImuFactor` and `CameraFeatureInterpFactor`
each reproject a frame's corners and report per-corner error norms. What they
share lives here:

* `_row_offsets()`: the rolling-shutter row offset rho of each measured corner,
  from its image row (all three factors);
* `_lam(calib, i)`: the interpolation fraction of corner i between the two
  bracketing states, (dt_frame + td + rho_i tr) / dt (the two interpolated factors);
* `_pB(h)`: the frame's board points with the estimated heights added along the
  board normal (the two board factors);
* `corner_norms(*args)`: per-corner norms (px) from the factor's `_residuals`,
  NaN where a guard skipped the corner;
* `corner_norms_values(values)`: the same at a gtsam Values, through the
  factor's `args_from_values` (keys from make_factor);
* `corner_errors(*args)` / `corner_errors_values(values)`: the signed (u, v)
  errors (px) behind those norms, rows of NaN where a guard skipped the corner.
"""

from __future__ import annotations

import numpy as np

from mvis.core.shutter import row_offset


class CornerReprojMixin:
    """Mixed into the reprojection factors; see the module docstring."""

    def _row_offsets(self):
        """rho of each corner in self.uv, from self.h_px and self.rolling_mode."""
        return np.array([row_offset(v, self.h_px, self.rolling_mode)
                         for v in self.uv[:, 1]])

    def _lam(self, calib, i: int) -> float:
        return (self.dt_frame + calib.td + self.rho[i] * calib.tr) / self.dt

    def _pB(self, h):
        """Board points of this frame's corners, with the estimated heights."""
        if self.pids is None or h is None:
            return self.p_B
        out = self.p_B.copy()
        out[:, 2] += np.asarray(h, float)[self.pids]
        return out

    def corner_norms(self, *args, **kwargs):
        """Per-corner reprojection error norms (px), Kalibr convention; NaN
        where a guard skipped the corner (excluded from stats, not zero).
        Takes the arguments of the factor's `error`."""
        return np.linalg.norm(self.corner_errors(*args, **kwargs), axis=1)

    def corner_norms_values(self, values):
        """corner_norms at a gtsam Values, using the keys from make_factor."""
        return self.corner_norms(*self.args_from_values(values))

    def corner_errors(self, *args, **kwargs):
        """Per-corner signed reprojection errors (n, 2) in px, (u, v) as the factor's
        `_residuals` gives them; rows of NaN where a guard skipped the corner. The
        norms of these rows are `corner_norms` bit for bit."""
        err, valid = self._residuals(*args, **kwargs)
        e = np.array(err, dtype=float).reshape(-1, 2)
        e[~np.asarray(valid, bool)] = np.nan
        return e

    def corner_errors_values(self, values):
        """corner_errors at a gtsam Values, using the keys from make_factor."""
        return self.corner_errors(*self.args_from_values(values))
