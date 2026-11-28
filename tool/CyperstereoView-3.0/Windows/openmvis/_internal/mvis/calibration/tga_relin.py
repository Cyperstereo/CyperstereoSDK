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
"""Move the IMU factors' t_ga linearization point when t_ga has travelled too far from it.

An IMU factor integrates once and corrects t_ga linearly, like a bias:
    Delta(t_ga) = Delta(t_lin) + J_tga * (t_ga - t_lin)
J_tga comes from the accel's slope, which changes a lot over several ms, so around 0 this is too
coarse for a large offset (T265: about -6 ms). When |t_ga - t_lin| passes a threshold, the IMU's
factors are re-integrated with the accel resampled at t + t_ga, which becomes the new t_lin.
"""
from __future__ import annotations

import logging
import time

import gtsam
import numpy as np

from mvis.calibration.factors.aci import ACIPreintegrator
from mvis.calibration.factors.imu_factor import ImuFactorACI_In
from mvis.calibration.factors.imu_factor_multi import ImuFactorACI_Multi_In_V
from mvis.calibration.lm_progress import IterationLog
from mvis.imu import create_imu_model

logger = logging.getLogger(__name__)


def shifted_accel(t, a, tau: float) -> np.ndarray:
    """The accel stream at t + tau: the accel that applies at gyro time t (see aci.py)."""
    if tau == 0.0:
        return a
    return np.column_stack([np.interp(t + tau, t, a[:, k]) for k in range(a.shape[1])])


class TgaStream:
    """One IMU's samples, its t_ga key, and the factors integrated from them."""

    def __init__(self, key, t, w, a, sigmas, model, name=""):
        self.key, self.name = key, name
        self.t, self.w, self.a = np.asarray(t, float), np.asarray(w, float), np.asarray(a, float)
        self.sigmas, self.model = sigmas, model
        self.t_lin = 0.0
        self.factors = []
        self.segments = []      # ImuSegments of the vision-IMU factors on this IMU

    def add(self, factor, t0: float, t1: float) -> None:
        self.factors.append((factor, t0, t1))

    def relinearize(self, tau: float) -> None:
        # Imported here: the pipeline module imports this one.
        from mvis.calibration.pipeline_cam_imu import _slice_imu
        a = shifted_accel(self.t, self.a, tau)
        for seg in self.segments:
            seg.relinearize_tga(a, tau)
        for factor, t0, t1 in self.factors:
            base = factor.base if isinstance(factor, ImuFactorACI_Multi_In_V) else factor
            pre = ACIPreintegrator(*self.sigmas, model=self.model, avg_meas=True)
            pre.reset(base.bg_init, base.ba_init)     # same bias point as before
            pre.set_intrinsics(create_imu_model(self.model))
            ts, ws, as_ = _slice_imu(self.t, self.w, a, t0, t1)
            pre.feed_imu_interval(ts, ws, as_)
            new = ImuFactorACI_In.from_preintegrator(pre, base.g_inG)
            new.t_lin = tau
            # In place: the graph's CustomFactor closures hold this object.
            base.__dict__.update(new.__dict__)
            if base is not factor:
                factor.w_inIs0 = np.asarray(pre.w_m_s, float)
                factor.w_inIs1 = np.asarray(pre.w_m_e, float)
        self.t_lin = tau


SKIP_ITERATIONS = 2


class TgaRelinOptimizer:
    """LM stepped one iteration at a time, with the API of the gtsam optimizer it replaces.

    After each iteration, any stream whose t_ga left the threshold is re-integrated. That changes
    the graph's error, so LM restarts from the current values, keeping its damping; the iteration
    is not tested for convergence. If nothing moves, this is gtsam's own loop.

    The first steps overshoot (twin: T265 to -9 ms, then back to -6 ms; other IMUs to -4 ms, then
    back to 0), so nothing moves in the first SKIP_ITERATIONS, unless the solve converges there."""

    #: optimize_logged hands its label to optimize() instead of stepping this itself.
    logs_progress = True

    def __init__(self, graph, values, params, streams, threshold_s: float):
        self.graph, self.params, self.streams = graph, params, streams
        self.threshold_s = float(threshold_s)
        self._opt = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
        self._iterations = 0
        self.relinearizations = []
        self._hold_td_refresh(True)

    def _hold_td_refresh(self, hold: bool) -> None:
        for s in self.streams:
            for seg in getattr(s, "segments", []):
                seg.allow_td_refresh = not hold

    def error(self) -> float:
        return self._opt.error()

    def iterations(self) -> int:
        return self._iterations

    def lambda_(self) -> float:
        return self._opt.lambda_()

    def _relinearize(self, values) -> bool:
        moved = False
        for s in self.streams:
            tau = float(values.atVector(s.key)[0])
            if abs(tau - s.t_lin) <= self.threshold_s:
                continue
            tic = time.time()
            previous = s.t_lin
            s.relinearize(tau)
            moved = True
            self.relinearizations.append(dict(
                imu=getattr(s, "name", ""), iteration=self._iterations, from_t_lin=previous,
                t_ga=tau, seconds=time.time() - tic))
            logger.info("t_ga relinearized: %s %.3f -> %.3f ms at iteration %d",
                        getattr(s, "name", ""), previous * 1e3, tau * 1e3, self._iterations)
        return moved

    def optimize(self, label: str = "cam-IMU LM"):
        p = self.params
        current = self._opt.error()
        log = IterationLog(p.getMaxIterations(), label)
        while self._iterations < p.getMaxIterations():
            before = self._opt.iterations()
            self._opt.iterate()
            # gtsam counts only accepted steps: a failed linear solve must stay at 0 iterations,
            # which is how _lm_no_step_report recognises it.
            stepped = self._opt.iterations() > before
            self._iterations += int(stepped)
            if self._iterations >= SKIP_ITERATIONS:
                self._hold_td_refresh(False)
            new = self._opt.error()
            log(self._iterations, current, new, self._opt.lambda_())
            values = self._opt.values()
            converged = gtsam.checkConvergence(p.getRelativeErrorTol(), p.getAbsoluteErrorTol(),
                                               p.getErrorTol(), current, new)
            if (self._iterations > SKIP_ITERATIONS or converged) and self._relinearize(values):
                p.setlambdaInitial(self._opt.lambda_())
                self._opt = gtsam.LevenbergMarquardtOptimizer(self.graph, values, p)
                current = self._opt.error()
                continue
            if converged or not stepped:
                break
            current = new
        return self._opt.values()
