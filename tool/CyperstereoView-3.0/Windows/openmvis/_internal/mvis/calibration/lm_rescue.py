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
"""Rescue a solve that LM gave up on far from a minimum.

From a poor start, one frame can hold nearly all the cost. LM then rejects steps until lambda reaches
its cap and stops, and that stop looks like convergence. The signal is the undamped step from the stop:
stuck solves were predicted to lose 98-99 % of their cost, converged ones at most 2e-6 of it (7 digital
twins and the good seeds of a hard scenario, 2026-11-02), or 11 % with per-frame time offsets, whose flat
directions promise more than a real step delivers.

Diagonal damping from the same start avoids such a point, but it is not safe as the only damping: it hit
the iteration limit on the mvis_elp twin with a lever 10x worse. So it only runs after a stuck stop, normal
damping then finishes, and a solve that did not stop stuck is not touched. The result is only kept if
its cost is lower.
"""
from __future__ import annotations

import logging

from mvis.calibration.lm_progress import optimize_logged

logger = logging.getLogger(__name__)

STUCK_RATIO = 0.5     # stuck solves 0.98; converged ones 2e-6, or 0.11 with per-frame time offsets
LAMBDA_RESTART = 1e-5  # GTSAM's initial lambda
ZERO_COST = 1e-12      # below this the ratio is round-off


def predicted_decrease_ratio(graph, values) -> float:
    """Share of the cost that the undamped linear step at `values` predicts to remove; 0 if the
    linear system cannot be solved, which `_lm_no_step_report` deals with."""
    gfg = graph.linearize(values)
    try:
        delta = gfg.optimize()
    except RuntimeError:                       # gtsam's IndeterminantLinearSystemException
        return 0.0
    cost = float(gfg.error(delta.scale(0.0)))
    return float(cost - gfg.error(delta)) / cost if cost > ZERO_COST else 0.0


class _Finished:
    """The last optimizer of a rescue, with the steps of every stage counted."""

    def __init__(self, opt, iterations: int):
        self._opt, self._iterations = opt, iterations

    def error(self) -> float:
        return self._opt.error()

    def iterations(self) -> int:
        return self._iterations


def rescue_if_stuck(graph, start, values, opt, params, make_opt, *, at_iteration_limit=False):
    """(values, opt, report). `report` is None unless LM stopped at its damping cap with most of the
    cost still predicted to go. Then it solves again from `start`, as the stuck point itself can be
    too far off to recover from. `make_opt(values, params)` builds an optimizer of the same kind.

    With `at_iteration_limit`, a stop at the iteration limit counts as well: from a poor start LM
    can still be descending, held at a large lambda, when its iterations run out. The camera-only
    solve of a fisheye camera seeded with a vanishing-point focal length about twice the real one
    ended there (Monado MGC05 cam0: 460.6 px after 15 iterations, 271.1 px after 39)."""
    capped = opt.lambda_() >= params.getlambdaUpperBound()
    limited = at_iteration_limit and opt.iterations() >= params.getMaxIterations()
    if not (capped or limited):
        return values, opt, None
    ratio = predicted_decrease_ratio(graph, values)
    if ratio < STUCK_RATIO:
        return values, opt, None
    if capped:
        logger.warning("LM gave up at its damping cap with %.0f %% of the cost still predicted to go; "
                       "rescuing with diagonal damping", ratio * 100)
    else:
        logger.warning("LM ran out of iterations (%d) with %.0f %% of the cost still predicted to go; "
                       "rescuing with diagonal damping", opt.iterations(), ratio * 100)
    stopped, stopped_opt = values, opt
    iterations, err_before = opt.iterations(), opt.error()
    values = start
    for diagonal in (True, False):
        params.setDiagonalDamping(diagonal)     # gtsam params cannot be copied; each stage ends first
        params.setlambdaInitial(LAMBDA_RESTART)
        opt = make_opt(values, params)
        values = optimize_logged(opt, params, "LM rescue")
        iterations += opt.iterations()
    err_rescued = float(opt.error())
    if err_rescued >= err_before:
        values, opt = stopped, stopped_opt
    after = predicted_decrease_ratio(graph, values)
    report = dict(ratio_before=ratio, ratio_after=after, err_before=float(err_before),
                  err_rescued=err_rescued, kept=bool(err_rescued < err_before),
                  failed=bool(after >= STUCK_RATIO))
    if report["failed"]:
        logger.warning("LM is still stuck after the rescue (%.0f %% of the cost predicted to go); "
                       "flagging the solve (lm_rescue.failed)", after * 100)
    return values, _Finished(opt, iterations), report
