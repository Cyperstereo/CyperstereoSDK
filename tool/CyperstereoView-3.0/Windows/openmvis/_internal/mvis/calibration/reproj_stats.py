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
"""Shared reprojection-error statistics — Kalibr convention.

Single source for BOTH pipelines (camera-only and cam-IMU, since
2026-08-29): a single reprojection error is the per-corner Euclidean norm in
raw pixels (unwhitened), and every reported statistic is computed over the
POOLED corner norms of all frames — per camera, or globally by concatenating
cameras. This matches Kalibr's printErrorStatistics population (all corners of
all views), so medians/p95s are directly comparable across tools.

Guard-skipped corners (cheirality / depth margin) enter as NaN and are
EXCLUDED from the population — never counted as zero-error.
"""

from __future__ import annotations

import numpy as np

_EMPTY = dict(rms=float("nan"), mean=float("nan"), median=float("nan"),
              std=float("nan"), p90=float("nan"), p95=float("nan"), max=float("nan"),
              n_corners=0)

# Corners within INLIER_SIGMAS x the solve's pixel sigma count as inliers for the
# second, Kalibr-comparable population (Kalibr reports after its own outlier removal).
INLIER_SIGMAS = 3.0


def corner_stats(norms) -> dict:
    """Statistics over pooled per-corner error norms (px).

    norms: array-like of per-corner Euclidean norms; NaN entries are excluded.
    Returns {rms, mean, median, std, p90, p95, max, n_corners}. std is the population
    std (ddof 0), as Kalibr prints it.
    """
    a = np.asarray(norms, float).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(_EMPTY)
    return dict(
        rms=float(np.sqrt(np.mean(a ** 2))),
        mean=float(np.mean(a)),
        median=float(np.median(a)),
        std=float(np.std(a)),
        p90=float(np.percentile(a, 90)),
        p95=float(np.percentile(a, 95)),
        max=float(np.max(a)),
        n_corners=int(a.size),
    )


def inlier_stats(norms, sigma_px: float, k: float = INLIER_SIGMAS) -> dict:
    """corner_stats over the corners within k * sigma_px, plus the threshold and how many
    finite corners it left out. The fixed, stated threshold makes the number reproducible;
    it stands in for Kalibr's outlier removal, which depends on its own batches."""
    a = np.asarray(norms, float).ravel()
    a = a[np.isfinite(a)]
    thr = float(k * sigma_px)
    s = corner_stats(a[a <= thr])
    s.update(threshold_px=thr, n_excluded=int(np.sum(a > thr)))
    return s


def reproj_entry(norms, sigma_px: float) -> dict:
    """The reprojection fields of a cam-IMU per-camera entry or of its overall diagnostics:
    reproj_{rms,mean,median,std,p90,p95,max}_px and reproj_n_corners over all corners, and
    reproj_inlier (inlier_stats). One builder for every pipeline that writes them."""
    s = corner_stats(norms)
    out = {f"reproj_{k}_px": s[k] for k in ("rms", "mean", "median", "std", "p90", "p95", "max")}
    out["reproj_n_corners"] = s["n_corners"]
    out["reproj_inlier"] = inlier_stats(norms, sigma_px)
    return out
