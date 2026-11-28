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
"""CLI entry point.

The first statement pins the BLAS thread count (default 1, `MVIS_BLAS_THREADS`
to change) — it must run before `mvis.cli.main` imports numpy / OpenCV / scipy,
see `mvis.cli.blas_threads`.  Library imports (`import mvis`, `mvis.core`, …)
never reach this package and keep their environment.
"""
from mvis.cli import blas_threads as _blas_threads

_blas_threads.STATUS = _blas_threads.pin_blas_threads()
