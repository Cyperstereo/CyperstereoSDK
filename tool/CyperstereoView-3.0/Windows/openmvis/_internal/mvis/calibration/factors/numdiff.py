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
"""Generic central-difference Jacobians for gtsam-typed residuals.

One helper to numerically differentiate any residual `fn(*args) -> vector` w.r.t.
each argument in its tangent space — perturbing `Pose3`/`Rot3` via `retract` and
plain `Vector` (ndarray) additively, auto-detected by type. Used to cross-check
the analytic factor Jacobians (replaces the per-factor `_numerical_jacobians` and
the finite-difference loops duplicated across the IMU tests).
"""

from __future__ import annotations

import numpy as np
import gtsam


def _tangent_dim(x) -> int:
    if isinstance(x, gtsam.Pose3):
        return 6
    if isinstance(x, gtsam.Rot3):
        return 3
    return len(np.asarray(x, float))


def _perturb(x, delta):
    if isinstance(x, (gtsam.Pose3, gtsam.Rot3)):
        return x.retract(delta)
    return np.asarray(x, float) + delta


def numerical_jacobian(fn, args, eps: float = 1e-6):
    """Central-difference Jacobian of ``fn(*args)`` w.r.t. each arg.

    Args:
        fn: callable returning a 1-D residual vector.
        args: list of variables — gtsam ``Pose3``/``Rot3`` (perturbed via
            ``retract``) or array-likes (perturbed additively).
        eps: finite-difference step.

    Returns:
        (e0, blocks): the residual at ``args`` and a list of Jacobian blocks,
        ``blocks[i]`` shaped ``(len(e0), tangent_dim(args[i]))``.
    """
    args = list(args)
    e0 = np.asarray(fn(*args), float)
    blocks = []
    for i, a in enumerate(args):
        n = _tangent_dim(a)
        J = np.zeros((e0.shape[0], n))
        for j in range(n):
            d = np.zeros(n); d[j] = eps
            ap = list(args); ap[i] = _perturb(a, d)
            am = list(args); am[i] = _perturb(a, -d)
            J[:, j] = (np.asarray(fn(*ap), float) - np.asarray(fn(*am), float)) / (2 * eps)
        blocks.append(J)
    return e0, blocks
