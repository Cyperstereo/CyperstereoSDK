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
"""Camera geometry for the report views, through MVIS's own forward model.

Every camera model goes through `project_observation`, the function the solver
and the residual report use, so a drawn ray or field-of-view rim agrees with
the calibration by construction. Unprojection inverts it numerically: no
per-model inverse is needed, and models without one (radtan, equidistant via
gtsam) work the same way.
"""
from __future__ import annotations

import math

import gtsam
import numpy as np

from mvis.calibration.factors.camera_projection import project_observation

_IDENTITY = gtsam.Pose3()


def project(model: str, cal_vec, X) -> np.ndarray | None:
    """Pixel of the camera-frame point X, or None when the model can't image it."""
    try:
        uv = project_observation(board_point=np.asarray(X, float), bp0=_IDENTITY, bp1=_IDENTITY,
                                 time_meas=0.0, time_0=0.0, time_1=1.0,
                                 cal_vec=np.asarray(cal_vec, float), distortion_model=model)
    except Exception:
        return None
    if uv is None or not np.all(np.isfinite(uv)):
        return None
    return np.asarray(uv, float)


def _dir(th: float, ph: float) -> np.ndarray:
    s = math.sin(th)
    return np.array([s * math.cos(ph), s * math.sin(ph), math.cos(th)])


def ray(model: str, cal_vec, uv, tol_px: float = 0.05, iters: int = 40) -> np.ndarray | None:
    """Unit ray (camera frame) that projects to pixel uv, by Gauss-Newton on the
    angle off the optical axis and the azimuth; None if it doesn't converge.

    Starts from the pinhole ray, so it handles fields of view past 180 degrees
    only when the model itself does."""
    cal = np.asarray(cal_vec, float)
    uv = np.asarray(uv, float)
    x, y = (uv[0] - cal[2]) / cal[0], (uv[1] - cal[3]) / cal[1]
    th, ph = math.atan(math.hypot(x, y)), math.atan2(y, x)
    eps = 1e-6
    for _ in range(iters):
        p = project(model, cal, _dir(th, ph))
        if p is None:
            th *= 0.8            # stepped past the model's domain: pull back toward the axis
            continue
        r = p - uv
        if float(np.hypot(*r)) < tol_px:
            return _dir(th, ph)
        pa = project(model, cal, _dir(th + eps, ph))
        pb = project(model, cal, _dir(th, ph + eps))
        if pa is None or pb is None:
            th *= 0.95
            continue
        J = np.column_stack([(pa - p) / eps, (pb - p) / eps])
        try:
            d = np.linalg.lstsq(J, -r, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
        th = float(np.clip(th + np.clip(d[0], -0.3, 0.3), 0.0, math.pi))
        ph = ph + float(np.clip(d[1], -0.5, 0.5))
    p = project(model, cal, _dir(th, ph))
    return _dir(th, ph) if p is not None and float(np.hypot(*(p - uv))) < 10 * tol_px else None


KB_MODELS = ("equidistant", "kb4")
KB_DOMAIN_DEG = 90.0   # MVIS projects these models only in front of the image plane (theta < 90 deg)


def _kb_ray(cal, uv, clamp: bool = False) -> np.ndarray | None:
    """Ray for a Kannala-Brandt pixel from the lens polynomial theta_d = theta (1 + k1 theta^2 + ... + k4 theta^8),
    inverted by bisection up to its turning point. Valid past 90 degrees, where MVIS's projection stops. With clamp, a
    pixel past the turning point gets the ray at the turning point (the edge as far as the model describes it)."""
    x, y = (uv[0] - cal[2]) / cal[0], (uv[1] - cal[3]) / cal[1]
    rd = math.hypot(x, y)
    if rd < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    k = cal[4:8]
    f = lambda t: t * (1 + t * t * (k[0] + t * t * (k[1] + t * t * (k[2] + t * t * k[3]))))
    hi, peak = math.pi, 0.0
    for i in range(1, 361):   # stay on the rising branch of the polynomial
        t = i * math.pi / 360
        v = f(t)
        if v < peak:
            hi = t - math.pi / 360
            break
        peak = v
        if v >= rd:
            hi = t
            break
    if f(hi) < rd:   # past the polynomial's turning point (a fold): this pixel has no angle in the model
        if not clamp:
            return None
        return np.array([math.sin(hi) * x / rd, math.sin(hi) * y / rd, math.cos(hi)])   # the fold, for drawing
    lo = 0.0
    for _ in range(60):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if f(m) < rd else (lo, m)
    th = 0.5 * (lo + hi)
    return np.array([math.sin(th) * x / rd, math.sin(th) * y / rd, math.cos(th)])


def edge_pixels(size, per_side: int = 8):
    """Pixels around the image border, clockwise from the top-left corner."""
    w, h = float(size[0]), float(size[1])
    out = []
    for k in range(per_side):
        out.append((w * k / per_side, 0.0))
    for k in range(per_side):
        out.append((w, h * k / per_side))
    for k in range(per_side):
        out.append((w - w * k / per_side, h))
    for k in range(per_side):
        out.append((0.0, h - h * k / per_side))
    return out


def _any_ray(model: str, cal_vec, uv, clamp: bool = False):
    """`ray`, falling back to the lens polynomial for Kannala-Brandt models past MVIS's 90 degree domain."""
    r = ray(model, cal_vec, uv)
    if r is None and model in KB_MODELS:
        r = _kb_ray(np.asarray(cal_vec, float), np.asarray(uv, float), clamp)
    return r


def rim(model: str, cal_vec, size, per_side: int = 8) -> list[list[float]]:
    """Unit rays through the image border: the field-of-view rim, camera frame.
    Border pixels the model can't invert are skipped."""
    rays = (_any_ray(model, cal_vec, uv, clamp=True) for uv in edge_pixels(size, per_side))
    return [[round(float(v), 5) for v in r] for r in rays if r is not None]


def fov_deg(model: str, cal_vec, size) -> dict:
    """Full horizontal and vertical field of view through the principal point, in degrees: the two edge rays'
    angles off the optical axis, added (so it can exceed 180 degrees). An axis is None when an edge has no ray,
    or, for Kannala-Brandt models, when an edge lies past the 90 degrees MVIS projects: past the fit."""
    cal = np.asarray(cal_vec, float)
    w, h = float(size[0]), float(size[1])

    def span(a, b):
        ra, rb = _any_ray(model, cal, a), _any_ray(model, cal, b)
        if ra is None or rb is None:
            return None
        ta, tb = (math.degrees(math.acos(float(np.clip(r[2], -1.0, 1.0)))) for r in (ra, rb))
        if model in KB_MODELS and max(ta, tb) > KB_DOMAIN_DEG:
            return None
        return round(ta + tb, 1)

    return {"h": span((0.0, cal[3]), (w, cal[3])), "v": span((cal[2], 0.0), (cal[2], h))}
