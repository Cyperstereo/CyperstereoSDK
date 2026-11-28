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
"""Pin the BLAS thread count of the `mvis` command line to 1 (speed-up step T0).

Why
---
The cam-IMU and cam-only solves are per-factor Python on 3×3 / 2×N matrices;
numpy's bundled OpenBLAS still spins one worker per core on every matmul.  On
the T7 bag 2 C reference (S0b, 32 cores) that costs ~590 s of `sys` time per
solve for no wall-time gain (default threads: wall 333-336 s, sys ~590 s;
1 thread: wall the same within the load noise, sys 2-4 s), and it makes the
result depend on the thread count (two identical runs differ at <= 1e-13 with
32 threads and are bitwise equal with 1).  So the CLI pins one thread by
default: same wall time, half the CPU per solve (parallel regression batches),
reproducible results.

Where
-----
`mvis.cli.__init__` calls `pin_blas_threads()` as its first statement, which
runs before `mvis.cli.main` imports numpy / OpenCV / scipy (the console script
`mvis`, `python -m mvis.cli.main`, and the regression runner's subprocesses
all enter there).  OpenBLAS reads the variables when its library is loaded,
so the pin must precede the first numpy import — this module imports only the
standard library.

Only the CLI pins.  `import mvis` as a library leaves the environment alone:
the `mvis` package is used from notebooks and other programs whose thread
policy is theirs (and the golden test pins in-process through
`tests/_blas_threads.py` where it needs 1 thread).  When numpy, scipy or cv2
are already imported — a test importing `mvis.cli.main`, `recalib.roundtrip`'s
lazy import, an embedding program — the pin could not take effect any more
and is skipped without touching the environment.

Override
--------
`MVIS_BLAS_THREADS=<n>`  pin to n threads (set explicitly, it is applied even
                         when OPENBLAS/OMP/MKL_NUM_THREADS are exported — it
                         is the mvis-specific request).
`MVIS_BLAS_THREADS=0`    (or empty) do not pin: OpenBLAS keeps its default.
unset                    pin to 1 unless one of OPENBLAS_NUM_THREADS,
                         GOTO_NUM_THREADS, OMP_NUM_THREADS or
                         MKL_NUM_THREADS is already exported: a value the
                         user set is never overridden (OpenBLAS honours
                         OMP_NUM_THREADS too, so an exported OMP value is a
                         thread choice even without OPENBLAS_NUM_THREADS).

The variables set are OPENBLAS_NUM_THREADS, OMP_NUM_THREADS and
MKL_NUM_THREADS (every BLAS a wheel may carry) and, with `setdefault`,
OPENBLAS_MAIN_FREE=1 (OpenBLAS must not change the main thread's CPU
affinity).  `mvis.cli.main` logs the outcome at INFO ("BLAS threads: …").
"""
import os
import sys

DEFAULT_THREADS = "1"
ENV_VAR = "MVIS_BLAS_THREADS"
#: variables this module sets
THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
#: an exported value in any of these is a user thread choice (OpenBLAS reads
#: OPENBLAS_, then GOTO_, then OMP_NUM_THREADS)
USER_VARS = THREAD_VARS + ("GOTO_NUM_THREADS",)
#: libraries that load a BLAS: once one is imported the pin cannot take effect
_BLAS_MODULES = ("numpy", "scipy", "cv2")

#: outcome of the CLI's pin (filled by `mvis.cli.__init__`); None until then
STATUS = None


def pin_blas_threads(environ=None, modules=None) -> dict:
    """Apply the policy above to `environ` (default `os.environ`).

    Returns a dict: `pinned` (bool), `threads` (int or None), `reason` (one
    short phrase), `set` (the variables written).  Never raises: a bad
    `MVIS_BLAS_THREADS` value is reported in `reason` and nothing is set.
    """
    environ = os.environ if environ is None else environ
    modules = sys.modules if modules is None else modules
    status = dict(pinned=False, threads=None, reason="", set=[])

    loaded = [m for m in _BLAS_MODULES if m in modules]
    if loaded:
        status["reason"] = (f"not pinned: {', '.join(loaded)} already imported "
                            f"before the CLI (library use)")
        return status

    raw = environ.get(ENV_VAR)
    explicit = raw is not None
    value = DEFAULT_THREADS if raw is None else raw.strip()
    if value in ("", "0"):
        status["reason"] = f"not pinned: {ENV_VAR}={raw!r} (BLAS default thread count)"
        return status
    try:
        n = int(value)
        if n < 0:
            raise ValueError
    except ValueError:
        status["reason"] = (f"not pinned: {ENV_VAR}={raw!r} is not a non-negative "
                            f"integer (BLAS default thread count)")
        return status

    if not explicit:
        exported = [v for v in USER_VARS if environ.get(v) not in (None, "")]
        if exported:
            v = exported[0]
            status["reason"] = (f"not pinned: {v}={environ[v]} exported by the user "
                                f"(set {ENV_VAR} to override)")
            return status

    for v in THREAD_VARS:
        environ[v] = str(n)
    status["set"] = list(THREAD_VARS)
    if environ.get("OPENBLAS_MAIN_FREE") is None:
        environ["OPENBLAS_MAIN_FREE"] = "1"
        status["set"].append("OPENBLAS_MAIN_FREE")
    status["pinned"] = True
    status["threads"] = n
    status["reason"] = (f"pinned to {n} ({ENV_VAR} explicit)" if explicit
                        else f"pinned to {n} (default; {ENV_VAR}=<n> or 0 to change)")
    return status


def describe(status=None) -> str:
    """One line for the CLI log, e.g. 'BLAS threads: pinned to 1 (default; …)'."""
    status = STATUS if status is None else status
    if status is None:
        return "BLAS threads: not pinned (pin_blas_threads() did not run)"
    return "BLAS threads: " + status["reason"]
