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
"""Make GTSAM's dense linear algebra independent of the core a solve starts on.

Why
---
Eigen sizes the blocks of its matrix products (GEMM, blocked QR / Cholesky)
from the CPU cache sizes, which it reads with CPUID once per process, on the
first large product, and keeps for the life of the process.  On a hybrid CPU
the answer depends on the core that runs that first product: on the
i9-14900HX of the development machine a P-core reports L1d 48 KiB / L2 2 MiB
and an E-core L1d 32 KiB / L2 4 MiB.  Other block sizes sum the dense
frontal matrices of the elimination in another order, so the same graph gives
results that differ in the last bits, and a solve is not repeatable: which
core the scheduler picks decides.  Measured on 2026-11-08:
monado_mgc/stereo01 pinned to a P-core gives pass-1 fx 269.98568948862805
in every run, pinned to an E-core 269.985689488628; unpinned it flips between
the two with the machine load.  A plain GTSAM bundle adjustment shows the same
split, and a process that makes its first product on a P-core and then moves
to an E-core keeps the P-core result (the choice is made once, at the first
product).  numpy's OpenBLAS and OpenCV give the same bytes on both core types.

What
----
`fix_eigen_cache_sizes()` pins the calling thread to ONE CPU of its allowed
set (the lowest-numbered one, so a given affinity mask always gives the same
core type), runs one small dense GTSAM elimination so that Eigen reads the
cache sizes there, and restores the thread's affinity.  It changes nothing in
what is computed: the block sizes are the ones Eigen would pick when the
process happened to start on that core.  It must run before the first GTSAM
solve of the process: a later call cannot change a choice already made, and
it cannot tell whether one was made, so callers make it their first GTSAM
use.  It runs once per process; later calls return the first status.

Where
-----
The solve entry points call it first: `mvis.cli.main.main` (every `mvis`
subcommand that solves) and `scripts/run_calibration.py`.  Library use
(`import mvis`) does not call it.

Override
--------
`MVIS_EIGEN_CACHE_CPU=<n>`  run the first product on CPU n (must be in the
                            allowed set).
`MVIS_EIGEN_CACHE_CPU=off`  do not pin (Eigen reads the cache sizes wherever
                            the first product runs, the old behaviour).
"""
from __future__ import annotations

import os

ENV_VAR = "MVIS_EIGEN_CACHE_CPU"

#: status of the first call in this process; None until then
STATUS: dict | None = None


def _warm_up() -> None:
    """One dense elimination large enough for Eigen's blocked kernels (a
    300 x 300 Jacobian factor: Householder QR with GEMM updates)."""
    import numpy as np
    import gtsam

    n = 300
    A = np.random.default_rng(0).standard_normal((n, n))
    g = gtsam.GaussianFactorGraph()
    g.add(0, A, np.ones(n), gtsam.noiseModel.Unit.Create(n))
    g.optimize()


def fix_eigen_cache_sizes(environ=None, warm_up=_warm_up) -> dict:
    """Apply the policy above once per process.  Returns a dict: `pinned`
    (bool), `cpu` (int or None), `reason` (one short phrase).  Never raises."""
    global STATUS
    if STATUS is not None:
        return STATUS
    environ = os.environ if environ is None else environ
    status = dict(pinned=False, cpu=None, reason="")
    if not hasattr(os, "sched_getaffinity"):
        status["reason"] = "not pinned: no sched_getaffinity on this platform"
        STATUS = status
        return status
    raw = (environ.get(ENV_VAR) or "").strip()
    if raw.lower() in ("off", "no", "false"):
        status["reason"] = f"not pinned: {ENV_VAR}={raw}"
        STATUS = status
        return status
    try:
        allowed = os.sched_getaffinity(0)
        cpu = int(raw) if raw else min(allowed)
        if cpu not in allowed:
            status["reason"] = (f"not pinned: {ENV_VAR}={raw} is not in the allowed "
                                f"CPU set")
            STATUS = status
            return status
        os.sched_setaffinity(0, {cpu})
        try:
            warm_up()
        finally:
            os.sched_setaffinity(0, allowed)
    except Exception as e:                    # never fail a solve over this
        status["reason"] = f"not pinned: {type(e).__name__}: {e}"
        STATUS = status
        return status
    status.update(pinned=True, cpu=cpu,
                  reason=f"cache sizes read on CPU {cpu}"
                         + (f" ({ENV_VAR})" if raw else " (lowest allowed CPU)"))
    STATUS = status
    return status


def describe(status=None) -> str:
    """One line for the solve log, e.g. 'Eigen cache sizes: read on CPU 0 …'."""
    status = STATUS if status is None else status
    if status is None:
        return "Eigen cache sizes: not fixed (fix_eigen_cache_sizes() did not run)"
    return "Eigen cache sizes: " + status["reason"]
