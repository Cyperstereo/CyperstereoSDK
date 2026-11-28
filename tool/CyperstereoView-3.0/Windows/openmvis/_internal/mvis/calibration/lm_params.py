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
"""One place that builds `gtsam.LevenbergMarquardtParams` for the mvis solvers.

GTSAM semantics that the defaults below rely on (verified in
`tests/calibration/test_lm_convergence.py` against the installed wrapper):

* After every accepted iteration LM checks `relative_decrease <= relative_error_tol`
  OR `absolute_decrease <= absolute_error_tol` and stops on the first hit. The
  returned values are the last *accepted* iterate, so lowering the tolerances
  only ever appends extra iterations (or extra rejected lambda retries) to the
  same prefix — it never changes the iterate a run would have stopped at.
* Inside one iteration a rejected step multiplies lambda by `lambda_factor` (10)
  and retries until `lambda >= lambda_upper` ("giving up because cannot decrease
  error with maximum lambda"), or until the attempted cost change is below
  `relative_error_tol * current_cost` (then it stops searching at once).
* Every accepted iteration divides lambda by the factor, floored at
  `lambda_lower`.

The cam-imu-speedup L1 step (2026-10-31) moved the camera-only CLI tolerances
from 1e-10 to the GTSAM default 1e-5 and capped lambda at 1e2, so a converged
bundle no longer burns up to 15 rejected ~45 s retries before giving up. The
number of tries a genuinely stuck iteration makes is the number of decades
between the lambda at the stall and `lambda_upper`: 7 from GTSAM's initial
1e-5, 12 from the 1e-10 a run reaches after five accepted iterations (the
board_rep stalls had 15 tries: 1e-10 .. 1e4 at GTSAM's 1e5). The lower bound
is left at GTSAM's 0 by default: any positive floor changes the damping of
every later accepted step (~1e-11 relative on the ELP-L bag-3 bundle), while
0 keeps every accepted iterate bitwise; `lm_lambda_lower: 1e-6` in the yaml
trades that for a <= 8-try cap on late stalls.

Covered sites: `optimizer.Optimizer` (camera-only; the CLI and the
`pipeline_cam.PipelineConfig` paths both feed it), the three LM solves in
`pipeline_cam_imu` and `multi_board_pipeline` (through `_lm_params`). The
Dogleg path is untouched.

The cap also ends a solve LM cannot improve from a poor start, which looks like
convergence: see `lm_rescue`.
"""

from __future__ import annotations

import gtsam

#: GTSAM defaults are lower 0.0 / upper 1e5; mvis keeps the lower bound (bitwise
#: results) and caps the upper one (retry loop).
LM_LAMBDA_LOWER_DEFAULT = 0.0
LM_LAMBDA_UPPER_DEFAULT = 1e2


def make_lm_params(
    max_iterations: int,
    *,
    absolute_error_tol: float | None = None,
    relative_error_tol: float | None = None,
    lambda_lower: float | None = LM_LAMBDA_LOWER_DEFAULT,
    lambda_upper: float | None = LM_LAMBDA_UPPER_DEFAULT,
    lambda_initial: float | None = None,
    verbose: bool = False,
) -> gtsam.LevenbergMarquardtParams:
    """Build LM params; `None` keeps the GTSAM default for that knob.

    `lambda_lower` / `lambda_upper` bound the damping (see module docstring);
    pass `None` (or 0 for the lower bound) to keep GTSAM's own 0 / 1e5.
    `lambda_initial` > 0 overrides the starting lambda (GTSAM 1e-5).
    `verbose` prints the per-try LM summary table (cost, lambda, time).
    """
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(int(max_iterations))
    if absolute_error_tol is not None:
        params.setAbsoluteErrorTol(float(absolute_error_tol))
    if relative_error_tol is not None:
        params.setRelativeErrorTol(float(relative_error_tol))
    if lambda_lower is not None:
        params.setlambdaLowerBound(float(lambda_lower))
    if lambda_upper is not None:
        params.setlambdaUpperBound(float(lambda_upper))
    if lambda_initial is not None and lambda_initial > 0:
        params.setlambdaInitial(float(lambda_initial))
    if verbose:
        params.setVerbosityLM("SUMMARY")
    return params
