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
"""Per-camera calibration memo for the CameraCalib vision factors (a speed-up).

Every evaluation of a reprojection factor over the `CameraCalib` bundle
(`CameraCalibInterpFactor`, `CameraCalibImuFactor`, and the test-only
`tests/helpers/camera_calib_factor.CameraCalibFactor`) used to
rebuild the same per-frame fixed setup from `(base, xi)`:

    calib    = base.retract(xi)                     CameraCalib (Pose3.retract + canonicalize)
    proj     = make_projector(calib.model, calib.intr)   CameraModel + gtsam Cal3
    Ad_TCinv = calib.T.inverse().AdjointMap()       jac path only
    Jr_xiT   = se3_right_jacobian(xi[0:6])          jac path only

Within one LM pass every factor of a camera sees the SAME xi (one calib key per
camera), so the ~1500 factors of a camera build ~1500 identical setups per pass.
`get_calib_setup` keeps one entry per base object and returns it while the key
matches; the two jac-only pieces are built lazily on first use.

Placement (plan amendment 5): NOT an attribute on the mutable `CameraCalib`
(pickling / deepcopy under `--parallel`, the sensor-clock two-pass reuse, and
staleness if T / intr / td / tr / model change). A module-level
`WeakKeyDictionary` keyed by the base object holds `(key, setup)` where

    key = (id(base), xi bytes, base.intr bytes, base.T matrix bytes,
           base.td, base.tr, base.model)

is the complete input of the setup, so a mutated base or a different xi misses.
A reused `id()` after garbage collection is harmless: the other fields pin the
inputs, and equal inputs give the identical setup. A `base.copy()` is a new
object with its own entry.

Bitwise: a hit returns the objects a miss would have built from the same
inputs by the same deterministic calls, so the factor numbers are unchanged
(`tests/cam_imu/test_speedup_k2.py` proves it against frozen state-dump oracles with a
cold and a warm memo). The returned setup is shared between callers and must
be treated as read-only.

Threading: gtsam's `CustomFactor` callbacks run sequentially in the local
(non-TBB) build; a TBB build may interleave them. The entry is ONE tuple
written atomically per base, and the key is compared on read, so an
interleaved reader either gets a setup whose key equals its own inputs or
rebuilds (never a mix of two setups). The lazy jac pieces are idempotent
(two writers store equal arrays).
"""

from __future__ import annotations

import weakref

import numpy as np

from .camera_calib import CameraCalib
from .lie_groups import se3_right_jacobian
from .projection import make_projector


class CalibSetup:
    """One camera's evaluated calibration at a tangent xi: `calib`, its
    projector, and (lazily) `Ad_TCinv` / `Jr_xiT` for the Jacobian path."""

    __slots__ = ("calib", "proj", "_xi6", "_Ad_TCinv", "_Jr_xiT")

    def __init__(self, calib: CameraCalib, proj, xi6: np.ndarray):
        self.calib = calib
        self.proj = proj
        self._xi6 = xi6
        self._Ad_TCinv = None
        self._Jr_xiT = None

    @property
    def Ad_TCinv(self) -> np.ndarray:
        """Ad(T_C^-1): maps the T_CtoW right tangent to the T_I right tangent."""
        A = self._Ad_TCinv
        if A is None:
            A = self._Ad_TCinv = self.calib.T.inverse().AdjointMap()
        return A

    @property
    def Jr_xiT(self) -> np.ndarray:
        """SE(3) right Jacobian of the retract chain around base.T."""
        J = self._Jr_xiT
        if J is None:
            J = self._Jr_xiT = se3_right_jacobian(self._xi6)
        return J


# base object -> (key, CalibSetup); one entry per base, replaced whole.
_memo: "weakref.WeakKeyDictionary[CameraCalib, tuple]" = weakref.WeakKeyDictionary()
_stats = {"hits": 0, "misses": 0}


def _key(base: CameraCalib, xi: np.ndarray) -> tuple:
    return (id(base), xi.tobytes(), base.intr.tobytes(), base.T.matrix().tobytes(),
            base.td, base.tr, base.model)


def get_calib_setup(base: CameraCalib, xi) -> CalibSetup:
    """The setup for `base.retract(xi)`: memoized per base while the key
    (xi and every base field the retract reads) is unchanged."""
    xi = np.asarray(xi, float)
    key = _key(base, xi)
    ent = _memo.get(base)
    if ent is not None and ent[0] == key:
        _stats["hits"] += 1
        return ent[1]
    _stats["misses"] += 1
    calib = base.retract(xi)
    setup = CalibSetup(calib, make_projector(calib.model, calib.intr), xi[0:6].copy())
    _memo[base] = (key, setup)
    return setup


def clear_calib_memo() -> None:
    """Drop every entry (tests / benchmarks)."""
    _memo.clear()
    _stats["hits"] = _stats["misses"] = 0


def calib_memo_stats() -> dict:
    """{'hits', 'misses', 'entries'} since the last clear."""
    return dict(_stats, entries=len(_memo))
