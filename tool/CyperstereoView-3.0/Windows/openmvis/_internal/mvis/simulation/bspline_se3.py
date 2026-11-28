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
"""Cumulative cubic B-spline on SE(3), reimplemented from the math.

A uniform cumulative cubic (order-4) B-spline over the SE(3) manifold, following
the standard construction (Sommer et al., "Efficient Derivative Computation for
Cumulative B-Splines on Lie Groups" — the same formulation OpenVINS' BsplineSE3
uses). Given uniformly time-spaced SE(3) control poses it returns, in closed
form, the pose and its first/second time derivatives, from which body angular
velocity, global linear velocity, and angular/linear acceleration are read off.
This is the trajectory generator under the camera-IMU simulator: IMU samples are
drawn from these analytic derivatives at the sensor rate.

From-the-math reimplementation (numpy + gtsam SE(3) exp/log), NOT a port of the
C++ source; the vehicle wheel / 2-D-motion constraints of MVIS'
`BsplineSE3Wheels` are intentionally omitted (not needed for cam-IMU calibration).

Conventions:
- A pose is the 4x4 homogeneous matrix ``T_ItoG = [[R_ItoG, p_IinG], [0, 1]]``.
- Tangent 6-vectors use gtsam ordering ``xi = [omega(3), rho(3)]`` (rotation
  first), consistent with ``gtsam.Pose3.Expmap/Logmap``.
- ``get_pose`` returns ``R_GtoI`` (= R_ItoG^T) and ``p_IinG``, matching the MVIS
  query signature; angular velocity ``w_IinI`` is in the body frame, linear
  velocity/acceleration ``v_IinG``/``a_IinG`` in the global frame.

For uniform cubic B-splines the cumulative basis on the segment ``u = (t-t1)/DT``
in ``[0, 1)`` (with bounding control poses indexed 0..3, t in [t1, t2]) is::

    b0 = (5 + 3u - 3u^2 +  u^3) / 6
    b1 = (1 + 3u + 3u^2 - 2u^3) / 6
    b2 = (              u^3) / 6

and ``T(t) = P0 * exp(b0 Ω1) * exp(b1 Ω2) * exp(b2 Ω3)`` where
``Ω_k = log(P_{k-1}^{-1} P_k)``. Derivatives follow by differentiating the basis.
"""

from __future__ import annotations

import numpy as np

from mvis.calibration.factors.lie_groups import (
    hat_se3 as _hat_se3,
    exp_se3 as _exp_se3,
    log_se3 as _log_se3,
    inv_se3 as _inv_se3,
    vee_so3 as _vee_so3,
)


class BsplineSE3:
    """Uniform cumulative cubic B-spline over SE(3).

    Parameters
    ----------
    t0 : float
        Time of the first control pose.
    dt : float
        Uniform spacing between control poses.
    control_poses : sequence of (4x4)
        Control poses ``T_ItoG``, uniformly spaced from ``t0`` by ``dt``. At
        least 4 are required; the queryable range is ``[t0 + dt, t0 + (N-2) dt]``
        (one control pose of padding is needed on each side).
    """

    def __init__(self, t0: float, dt: float, control_poses):
        """Store the uniformly spaced control poses. Requires >= 4 (the cumulative
        cubic basis needs one pose of padding on each side); raises `ValueError`
        otherwise."""
        self.t0 = float(t0)
        self.dt = float(dt)
        self.ctrl = [np.asarray(P, float) for P in control_poses]
        if len(self.ctrl) < 4:
            raise ValueError("B-spline needs >= 4 control poses")

    # ------------------------------------------------------------------
    @classmethod
    def from_pose_fn(cls, pose_fn, t0: float, t1: float, dt: float):
        """Build control poses by sampling an analytic ``pose_fn(t) -> T_ItoG``
        (4x4) at uniform spacing ``dt`` over ``[t0 - dt, t1 + dt]`` (with padding
        so the whole ``[t0, t1]`` interval is queryable)."""
        n = int(np.ceil((t1 - t0) / dt)) + 3
        ts = t0 - dt + np.arange(n) * dt
        return cls(ts[0], dt, [np.asarray(pose_fn(t), float) for t in ts])

    @property
    def start_time(self) -> float:
        """First queryable time — one control pose in from `t0` (the padding knot)."""
        return self.t0 + self.dt

    @property
    def end_time(self) -> float:
        """Last queryable time — one control pose short of the end, mirroring
        `start_time`. Querying outside `[start_time, end_time]` is not valid."""
        return self.t0 + (len(self.ctrl) - 2) * self.dt

    def _segment(self, t: float):
        """Return (P0, P1, P2, P3, u) — the 4 bounding control poses and the
        normalized segment coordinate u in [0, 1) for query time t."""
        seg = int(np.floor((t - self.t0) / self.dt))
        seg = max(1, min(seg, len(self.ctrl) - 3))    # clamp so 0..3 exist
        u = (t - (self.t0 + seg * self.dt)) / self.dt
        c = self.ctrl
        return c[seg - 1], c[seg], c[seg + 1], c[seg + 2], u

    def _eval(self, t: float, order: int = 2):
        """Return (T, Tdot, Tddot) 4x4 matrices; higher orders are None when not
        requested (order 0/1/2)."""
        P0, P1, P2, P3, u = self._segment(t)
        DT = self.dt

        O1 = _log_se3(_inv_se3(P0) @ P1)
        O2 = _log_se3(_inv_se3(P1) @ P2)
        O3 = _log_se3(_inv_se3(P2) @ P3)
        H1, H2, H3 = _hat_se3(O1), _hat_se3(O2), _hat_se3(O3)

        b0 = (5 + 3 * u - 3 * u ** 2 + u ** 3) / 6.0
        b1 = (1 + 3 * u + 3 * u ** 2 - 2 * u ** 3) / 6.0
        b2 = (u ** 3) / 6.0
        A0, A1, A2 = _exp_se3(b0 * O1), _exp_se3(b1 * O2), _exp_se3(b2 * O3)

        T = P0 @ A0 @ A1 @ A2
        if order == 0:
            return T, None, None

        b0d = (3 - 6 * u + 3 * u ** 2) / (6.0 * DT)
        b1d = (3 + 6 * u - 6 * u ** 2) / (6.0 * DT)
        b2d = (3 * u ** 2) / (6.0 * DT)
        A0d = b0d * H1 @ A0
        A1d = b1d * H2 @ A1
        A2d = b2d * H3 @ A2
        Td = P0 @ (A0d @ A1 @ A2 + A0 @ A1d @ A2 + A0 @ A1 @ A2d)
        if order == 1:
            return T, Td, None

        b0dd = (-6 + 6 * u) / (6.0 * DT ** 2)
        b1dd = (6 - 12 * u) / (6.0 * DT ** 2)
        b2dd = (6 * u) / (6.0 * DT ** 2)
        A0dd = b0dd * H1 @ A0 + b0d * H1 @ A0d
        A1dd = b1dd * H2 @ A1 + b1d * H2 @ A1d
        A2dd = b2dd * H3 @ A2 + b2d * H3 @ A2d
        Tdd = P0 @ (A0dd @ A1 @ A2 + A0 @ A1dd @ A2 + A0 @ A1 @ A2dd
                    + 2 * A0d @ A1d @ A2 + 2 * A0d @ A1 @ A2d
                    + 2 * A0 @ A1d @ A2d)
        return T, Td, Tdd

    # ------------------------------------------------------------------
    def get_pose(self, t: float):
        """Return (R_GtoI, p_IinG)."""
        T, _, _ = self._eval(t, 0)
        R_ItoG = T[0:3, 0:3]
        return R_ItoG.T, T[0:3, 3].copy()

    def get_velocity(self, t: float):
        """Return (R_GtoI, p_IinG, w_IinI, v_IinG)."""
        T, Td, _ = self._eval(t, 1)
        R_ItoG = T[0:3, 0:3]
        w = _vee_so3(R_ItoG.T @ Td[0:3, 0:3])
        return R_ItoG.T, T[0:3, 3].copy(), w, Td[0:3, 3].copy()

    def get_acceleration(self, t: float):
        """Return (R_GtoI, p_IinG, w_IinI, v_IinG, alpha_IinI, a_IinG)."""
        T, Td, Tdd = self._eval(t, 2)
        R_ItoG = T[0:3, 0:3]
        wsk = R_ItoG.T @ Td[0:3, 0:3]
        w = _vee_so3(wsk)
        alpha_sk = R_ItoG.T @ Tdd[0:3, 0:3] - wsk @ wsk
        alpha = _vee_so3(0.5 * (alpha_sk - alpha_sk.T))   # skew-symmetric part
        return (R_ItoG.T, T[0:3, 3].copy(), w, Td[0:3, 3].copy(),
                alpha, Tdd[0:3, 3].copy())
