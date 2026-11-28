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
"""Progress lines for long Levenberg–Marquardt solves.

``gtsam.LevenbergMarquardtOptimizer.optimize()`` runs every iteration inside one
C++ call: a cam-IMU solve of 4 cameras × 1,075 frames printed nothing for over
half an hour. ``optimize_logged`` runs the same loop from Python, one
``iterate()`` at a time, and logs an iteration line whenever ``interval_s`` has
passed since the last one, so short solves stay quiet and long ones report.

The loop is GTSAM 4.2's ``NonlinearOptimizer::defaultOptimize`` step for step
(error tolerance, iteration cap, ``checkConvergence`` on consecutive errors,
finite-error guard), so the result is bitwise the one ``optimize()`` returns
(tests/calibration/test_lm_progress.py pins it).
"""
from __future__ import annotations

import logging
import math
import os
import time

import gtsam

logger = logging.getLogger("mvis.calibration.lm_progress")

__all__ = ["LM_PROGRESS_INTERVAL_S", "IterationLog", "optimize_logged"]

#: Seconds between LM iteration lines; ``MVIS_LM_PROGRESS_S`` overrides, 0 = off.
LM_PROGRESS_INTERVAL_S = 10.0


def _interval_s() -> float:
    raw = os.environ.get("MVIS_LM_PROGRESS_S")
    if raw is None:
        return LM_PROGRESS_INTERVAL_S
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning(f"MVIS_LM_PROGRESS_S={raw!r} is not a number; "
                       f"using {LM_PROGRESS_INTERVAL_S:g} s")
        return LM_PROGRESS_INTERVAL_S


class IterationLog:
    """Throttled ``LM it k/N: error a -> b, lambda l, s/it`` lines."""

    def __init__(self, max_iterations: int, label: str = "LM",
                 interval_s: float | None = None, clock=time.monotonic):
        self._max = int(max_iterations)
        self._label = label
        self._interval = _interval_s() if interval_s is None else interval_s
        self._clock = clock
        self._t0 = self._last = clock()

    def __call__(self, iteration: int, err_before: float, err_after: float,
                 lam: float | None = None) -> None:
        if self._interval <= 0:
            return
        now = self._clock()
        if now - self._last < self._interval:
            return
        self._last = now
        per_it = (now - self._t0) / max(iteration, 1)
        lam_s = f", lambda {lam:.3g}" if lam is not None else ""
        logger.info(f"  {self._label} it {iteration}/{self._max}: error "
                    f"{err_before:.6g} -> {err_after:.6g}{lam_s}, {per_it:.1f} s/it")


def optimize_logged(opt, params, label: str = "LM"):
    """``opt.optimize()`` with throttled per-iteration progress lines.

    A plain ``gtsam.LevenbergMarquardtOptimizer`` runs GTSAM's own loop here,
    iterate by iterate. An optimizer that steps and logs itself (``logs_progress``,
    e.g. ``TgaRelinOptimizer``) gets ``opt.optimize(label)``; anything else
    (e.g. Dogleg) is just ``opt.optimize()`` and ``params`` is not read."""
    if getattr(opt, "logs_progress", False):
        return opt.optimize(label)
    if not isinstance(opt, gtsam.LevenbergMarquardtOptimizer):
        return opt.optimize()
    log = IterationLog(params.getMaxIterations(), label)
    max_it = params.getMaxIterations()
    rel, abs_, err_tol = (params.getRelativeErrorTol(), params.getAbsoluteErrorTol(),
                          params.getErrorTol())
    current = opt.error()
    if current <= err_tol or opt.iterations() >= max_it:      # defaultOptimize's early outs
        return opt.values()
    new = current
    while True:
        current = new
        opt.iterate()
        new = opt.error()
        log(opt.iterations(), current, new, opt.lambda_())
        if not (opt.iterations() < max_it
                and not gtsam.checkConvergence(rel, abs_, err_tol, current, new)
                and math.isfinite(current)):
            break
    return opt.values()
