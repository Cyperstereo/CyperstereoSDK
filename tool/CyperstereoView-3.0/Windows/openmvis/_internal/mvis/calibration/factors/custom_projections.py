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
"""Custom camera projection functions with analytical Jacobians.

Implements FOV, Double Sphere, EUCM, UCM (alpha form: the EUCM kernel at
beta = 1, with Basalt's fold check), omni-radtan and Kannala-Brandt KB-N (kb4)
models that are not available in GTSAM. Each function returns
(pixel, J_point, J_cal) where:
  - pixel: [u, v] or None if point is behind camera
  - J_point: (2, 3) Jacobian w.r.t. [X, Y, Z] in camera frame
  - J_cal: (2, max(8, len(cal_vec))) Jacobian w.r.t. cal_vec
    [fx, fy, cx, cy, d0, d1, ...] (8 wide for the zero-padded camera-only K;
    9 for omni-radtan with p2)

Math reference: doc/camera_models.html sections 3, 4, 5.
"""

import math
from functools import lru_cache

import numpy as np

from mvis.core.camera_models import CUSTOM


# ---------------------------------------------------------------------------
# FOV model (Devernay & Faugeras 2001)
# cal_vec: [fx, fy, cx, cy, omega, 0, 0, 0]
# ---------------------------------------------------------------------------

def project_fov(
    p_cam: np.ndarray,
    cal_vec: np.ndarray,
    compute_jacobians: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """FOV model projection.

    Forward:
        x = X/Z, y = Y/Z, r_u = sqrt(x² + y²)
        r_d = (1/omega) * arctan(2 * r_u * tan(omega/2))
        s = r_d / r_u  (scaling factor)
        u = fx * s * x + cx,  v = fy * s * y + cy
    """
    X, Y, Z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
    if Z <= 1e-8:
        return None, None, None

    fx, fy, cx, cy = cal_vec[0], cal_vec[1], cal_vec[2], cal_vec[3]
    omega = cal_vec[4]

    x = X / Z
    y = Y / Z
    r_u = np.sqrt(x * x + y * y)

    # Scaling factor s = r_d / r_u
    if abs(omega) < 1e-8:
        # Pinhole limit: s → 1
        s = 1.0
        ds_dru = 0.0
        ds_domega = 0.0
    else:
        tan_half = np.tan(omega / 2.0)
        if r_u < 1e-10:
            # On optical axis: s = (2*tan(omega/2)) / omega via L'Hôpital
            s = 2.0 * tan_half / omega
            ds_dru = 0.0
            ds_domega = 0.0  # simplified at axis
        else:
            arg = 2.0 * r_u * tan_half
            r_d = np.arctan(arg) / omega
            s = r_d / r_u

            # ds/d(r_u): d(r_d/r_u)/d(r_u) = (dr_d/dr_u - s) / r_u
            # dr_d/dr_u = 2*tan_half / (omega * (1 + arg²))
            dr_d_dru = 2.0 * tan_half / (omega * (1.0 + arg * arg))
            ds_dru = (dr_d_dru - s) / r_u

            # ds/d(omega): d(r_d/r_u)/d(omega) = (1/r_u) * dr_d/domega
            # dr_d/domega = -arctan(arg)/omega² + (r_u / (omega*(1+arg²))) * sec²(omega/2)
            sec_half_sq = 1.0 / (np.cos(omega / 2.0) ** 2)
            dr_d_domega = (-np.arctan(arg) / (omega * omega)
                           + r_u * sec_half_sq / (omega * (1.0 + arg * arg)))
            ds_domega = dr_d_domega / r_u

    x_d = s * x
    y_d = s * y
    u = fx * x_d + cx
    v = fy * y_d + cy
    pixel = np.array([u, v])

    if not compute_jacobians:
        return pixel, None, None

    # --- Jacobian w.r.t. [X, Y, Z] ---
    # x = X/Z, y = Y/Z
    # dx/dX = 1/Z, dx/dZ = -X/Z², dy/dY = 1/Z, dy/dZ = -Y/Z²
    # r_u = sqrt(x²+y²), dr_u/dx = x/r_u, dr_u/dy = y/r_u
    # s = s(r_u)
    # x_d = s*x, y_d = s*y
    # dx_d/dx = s + x * ds/dx = s + x * ds_dru * x/r_u
    # dx_d/dy = x * ds_dru * y/r_u
    # similarly for y_d

    if r_u < 1e-10:
        # At optical axis, simplified
        dxd_dx = s
        dxd_dy = 0.0
        dyd_dx = 0.0
        dyd_dy = s
    else:
        dxd_dx = s + x * ds_dru * x / r_u
        dxd_dy = x * ds_dru * y / r_u
        dyd_dx = y * ds_dru * x / r_u
        dyd_dy = s + y * ds_dru * y / r_u

    # d(x,y)/d(X,Y,Z)
    # dx/dX = 1/Z, dx/dY = 0, dx/dZ = -X/Z²
    # dy/dX = 0, dy/dY = 1/Z, dy/dZ = -Y/Z²
    inv_Z = 1.0 / Z

    # Chain: d(x_d)/d(X) = dxd_dx * dx/dX + dxd_dy * dy/dX
    J_xd = np.array([
        [dxd_dx * inv_Z, dxd_dy * inv_Z, -(dxd_dx * x + dxd_dy * y) * inv_Z],
        [dyd_dx * inv_Z, dyd_dy * inv_Z, -(dyd_dx * x + dyd_dy * y) * inv_Z],
    ])
    J_point = np.array([[fx, 0], [0, fy]]) @ J_xd  # (2,3)

    # --- Jacobian w.r.t. cal_vec [fx, fy, cx, cy, omega, 0, 0, 0] ---
    J_cal = np.zeros((2, max(8, len(cal_vec))))
    J_cal[0, 0] = x_d      # du/dfx
    J_cal[1, 1] = y_d      # dv/dfy
    J_cal[0, 2] = 1.0      # du/dcx
    J_cal[1, 3] = 1.0      # dv/dcy
    # du/domega = fx * x * ds/domega
    # dv/domega = fy * y * ds/domega
    J_cal[0, 4] = fx * x * ds_domega
    J_cal[1, 4] = fy * y * ds_domega

    return pixel, J_point, J_cal


# ---------------------------------------------------------------------------
# Double Sphere model (Usenko et al. 2018)
# cal_vec: [fx, fy, cx, cy, xi, alpha, 0, 0]
# ---------------------------------------------------------------------------

def project_double_sphere(
    p_cam: np.ndarray,
    cal_vec: np.ndarray,
    compute_jacobians: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Double Sphere model projection.

    Forward:
        d1 = ||P||
        d2 = sqrt(X² + Y² + (xi*d1 + Z)²)
        w = alpha * d2 + (1-alpha) * (xi*d1 + Z)
        u = fx * X/w + cx,  v = fy * Y/w + cy

    Validity: w > 0
    """
    X, Y, Z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
    fx, fy, cx, cy = cal_vec[0], cal_vec[1], cal_vec[2], cal_vec[3]
    xi = cal_vec[4]
    alpha = cal_vec[5]

    d1 = np.sqrt(X * X + Y * Y + Z * Z)
    if d1 < 1e-10:
        return None, None, None

    zs = xi * d1 + Z  # shifted Z
    d2 = np.sqrt(X * X + Y * Y + zs * zs)
    if d2 < 1e-10:
        return None, None, None

    w = alpha * d2 + (1.0 - alpha) * zs
    if w <= 1e-10:
        return None, None, None

    inv_w = 1.0 / w
    u = fx * X * inv_w + cx
    v = fy * Y * inv_w + cy
    pixel = np.array([u, v])

    if not compute_jacobians:
        return pixel, None, None

    # --- Jacobian w.r.t. [X, Y, Z] ---
    # d(d1)/d(X) = X/d1, d(d1)/d(Y) = Y/d1, d(d1)/d(Z) = Z/d1
    inv_d1 = 1.0 / d1
    inv_d2 = 1.0 / d2
    inv_w2 = inv_w * inv_w

    # d(zs)/d(X) = xi * X/d1, d(zs)/d(Y) = xi * Y/d1, d(zs)/d(Z) = xi * Z/d1 + 1
    dzs_dX = xi * X * inv_d1
    dzs_dY = xi * Y * inv_d1
    dzs_dZ = xi * Z * inv_d1 + 1.0

    # d(d2)/d(X) = (X + zs * dzs_dX) / d2
    dd2_dX = (X + zs * dzs_dX) * inv_d2
    dd2_dY = (Y + zs * dzs_dY) * inv_d2
    dd2_dZ = (zs * dzs_dZ) * inv_d2

    # d(w)/d(X) = alpha * dd2_dX + (1-alpha) * dzs_dX
    dw_dX = alpha * dd2_dX + (1.0 - alpha) * dzs_dX
    dw_dY = alpha * dd2_dY + (1.0 - alpha) * dzs_dY
    dw_dZ = alpha * dd2_dZ + (1.0 - alpha) * dzs_dZ

    # d(u)/d(X) = fx * (w - X * dw_dX) / w²
    # d(u)/d(Y) = fx * (-X * dw_dY) / w²
    # d(u)/d(Z) = fx * (-X * dw_dZ) / w²
    J_point = np.array([
        [fx * (w - X * dw_dX) * inv_w2,
         fx * (-X * dw_dY) * inv_w2,
         fx * (-X * dw_dZ) * inv_w2],
        [fy * (-Y * dw_dX) * inv_w2,
         fy * (w - Y * dw_dY) * inv_w2,
         fy * (-Y * dw_dZ) * inv_w2],
    ])

    # --- Jacobian w.r.t. cal_vec [fx, fy, cx, cy, xi, alpha, 0, 0] ---
    J_cal = np.zeros((2, max(8, len(cal_vec))))
    J_cal[0, 0] = X * inv_w   # du/dfx
    J_cal[1, 1] = Y * inv_w   # dv/dfy
    J_cal[0, 2] = 1.0         # du/dcx
    J_cal[1, 3] = 1.0         # dv/dcy

    # du/dxi: need dw/dxi
    # zs = xi*d1 + Z, d(zs)/d(xi) = d1
    # d(d2)/d(xi) = zs * d1 / d2
    # d(w)/d(xi) = alpha * zs*d1/d2 + (1-alpha) * d1
    dw_dxi = alpha * zs * d1 * inv_d2 + (1.0 - alpha) * d1
    J_cal[0, 4] = fx * (-X * dw_dxi) * inv_w2
    J_cal[1, 4] = fy * (-Y * dw_dxi) * inv_w2

    # du/dalpha: dw/dalpha = d2 - zs
    dw_dalpha = d2 - zs
    J_cal[0, 5] = fx * (-X * dw_dalpha) * inv_w2
    J_cal[1, 5] = fy * (-Y * dw_dalpha) * inv_w2

    return pixel, J_point, J_cal


# ---------------------------------------------------------------------------
# EUCM model (Khomutenko et al. 2016)
# cal_vec: [fx, fy, cx, cy, alpha, beta, 0, 0]
# ---------------------------------------------------------------------------

def project_eucm(
    p_cam: np.ndarray,
    cal_vec: np.ndarray,
    compute_jacobians: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Extended Unified Camera Model projection.

    Forward:
        d = sqrt(beta*(X² + Y²) + Z²)
        w = alpha * d + (1-alpha) * Z
        u = fx * X/w + cx,  v = fy * Y/w + cy
    """
    X, Y, Z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
    fx, fy, cx, cy = cal_vec[0], cal_vec[1], cal_vec[2], cal_vec[3]
    alpha = cal_vec[4]
    beta = cal_vec[5]

    rho_sq = X * X + Y * Y
    d = np.sqrt(beta * rho_sq + Z * Z)
    if d < 1e-10:
        return None, None, None

    w = alpha * d + (1.0 - alpha) * Z
    if w <= 1e-10:
        return None, None, None

    inv_w = 1.0 / w
    u = fx * X * inv_w + cx
    v = fy * Y * inv_w + cy
    pixel = np.array([u, v])

    if not compute_jacobians:
        return pixel, None, None

    # --- Jacobian w.r.t. [X, Y, Z] ---
    inv_d = 1.0 / d
    inv_w2 = inv_w * inv_w

    # d(d)/d(X) = beta*X / d
    dd_dX = beta * X * inv_d
    dd_dY = beta * Y * inv_d
    dd_dZ = Z * inv_d

    # d(w)/d(X) = alpha * dd_dX
    dw_dX = alpha * dd_dX
    dw_dY = alpha * dd_dY
    dw_dZ = alpha * dd_dZ + (1.0 - alpha)

    J_point = np.array([
        [fx * (w - X * dw_dX) * inv_w2,
         fx * (-X * dw_dY) * inv_w2,
         fx * (-X * dw_dZ) * inv_w2],
        [fy * (-Y * dw_dX) * inv_w2,
         fy * (w - Y * dw_dY) * inv_w2,
         fy * (-Y * dw_dZ) * inv_w2],
    ])

    # --- Jacobian w.r.t. cal_vec [fx, fy, cx, cy, alpha, beta, 0, 0] ---
    J_cal = np.zeros((2, max(8, len(cal_vec))))
    J_cal[0, 0] = X * inv_w   # du/dfx
    J_cal[1, 1] = Y * inv_w   # dv/dfy
    J_cal[0, 2] = 1.0         # du/dcx
    J_cal[1, 3] = 1.0         # dv/dcy

    # du/dalpha: dw/dalpha = d - Z
    dw_dalpha = d - Z
    J_cal[0, 4] = fx * (-X * dw_dalpha) * inv_w2
    J_cal[1, 4] = fy * (-Y * dw_dalpha) * inv_w2

    # du/dbeta: dw/dbeta = alpha * d(d)/d(beta)
    # d(d)/d(beta) = rho_sq / (2*d)
    dd_dbeta = rho_sq / (2.0 * d)
    dw_dbeta = alpha * dd_dbeta
    J_cal[0, 5] = fx * (-X * dw_dbeta) * inv_w2
    J_cal[1, 5] = fy * (-Y * dw_dbeta) * inv_w2

    return pixel, J_point, J_cal


def ucm_fold_w(alpha: float) -> float:
    """Basalt UnifiedCamera validity factor: a point is projectable iff
    Z > -w * |X|. Past that the image folds back (alpha = 0.64: ~124 deg)."""
    if alpha > 0.5:
        return (1.0 - alpha) / alpha
    return alpha / (1.0 - alpha)


def project_ucm(
    p_cam: np.ndarray,
    cal_vec: np.ndarray,
    compute_jacobians: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Unified Camera Model projection, alpha form (Basalt `ucm`; EUCM with beta=1).

    Forward:
        d = sqrt(X² + Y² + Z²)
        w = alpha * d + (1-alpha) * Z
        u = fx * X/w + cx,  v = fy * Y/w + cy
    Valid iff Z > -w_fold * d (`ucm_fold_w`). Evaluated by `project_eucm` at
    [fx, fy, cx, cy, alpha, 1]: only cal_vec[:5] is read, so a zero- or
    garbage-padded cal_vec never feeds a beta. J_cal is 2 x max(8, len(cal_vec))
    with columns 5.. zero.
    """
    alpha = float(cal_vec[4])
    X, Y, Z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
    if Z <= -ucm_fold_w(alpha) * math.sqrt(X * X + Y * Y + Z * Z):
        return None, None, None
    cal8 = np.array([cal_vec[0], cal_vec[1], cal_vec[2], cal_vec[3], alpha, 1.0, 0.0, 0.0])
    pixel, J_point, J_eucm = project_eucm(p_cam, cal8, compute_jacobians)
    if pixel is None or not compute_jacobians:
        return pixel, None, None
    J_cal = np.zeros((2, max(8, len(cal_vec))))
    J_cal[:, :5] = J_eucm[:, :5]
    return pixel, J_point, J_cal


# ---------------------------------------------------------------------------
# Unprojection functions (closed-form for all three models)
# ---------------------------------------------------------------------------

def unproject_fov(pixel: np.ndarray, cal_vec: np.ndarray) -> np.ndarray | None:
    """FOV model unprojection → unit ray [X, Y, Z]."""
    fx, fy, cx, cy = cal_vec[0], cal_vec[1], cal_vec[2], cal_vec[3]
    omega = cal_vec[4]

    x_d = (pixel[0] - cx) / fx
    y_d = (pixel[1] - cy) / fy
    r_d = np.sqrt(x_d * x_d + y_d * y_d)

    if abs(omega) < 1e-8 or r_d < 1e-10:
        return np.array([x_d, y_d, 1.0])

    tan_half = np.tan(omega / 2.0)
    r_u = np.tan(r_d * omega) / (2.0 * tan_half)
    s = r_u / r_d
    x = s * x_d
    y = s * y_d
    return np.array([x, y, 1.0])


def unproject_double_sphere(pixel: np.ndarray, cal_vec: np.ndarray) -> np.ndarray | None:
    """Double Sphere unprojection → unit ray [X, Y, Z]."""
    fx, fy, cx, cy = cal_vec[0], cal_vec[1], cal_vec[2], cal_vec[3]
    xi = cal_vec[4]
    alpha = cal_vec[5]

    mx = (pixel[0] - cx) / fx
    my = (pixel[1] - cy) / fy
    r2 = mx * mx + my * my

    disc = 1.0 - (2.0 * alpha - 1.0) * r2
    if disc <= 0:
        return None

    mz = (1.0 - alpha * alpha * r2) / (alpha * np.sqrt(disc) + 1.0 - alpha)
    denom = mz * mz + r2
    if denom < 1e-10:
        return None

    inner = mz * mz + (1.0 - xi * xi) * r2
    if inner < 0:
        return None

    lam = (mz * xi + np.sqrt(inner)) / denom
    return np.array([lam * mx, lam * my, lam * mz - xi])


def unproject_eucm(pixel: np.ndarray, cal_vec: np.ndarray) -> np.ndarray | None:
    """EUCM unprojection → unit ray [X, Y, Z].

    Derivation: given mx=(u-cx)/fx, my=(v-cy)/fy, r²=mx²+my²,
    we have X=mx*w, Y=my*w, Z=k*w where k=Z/w.
    From w = alpha*d + (1-alpha)*Z and d = sqrt(beta*r²*w² + Z²):
      1 = alpha*sqrt(beta*r² + k²) + (1-alpha)*k
    Solving the quadratic for k gives the ray direction.
    """
    fx, fy, cx, cy = cal_vec[0], cal_vec[1], cal_vec[2], cal_vec[3]
    alpha = cal_vec[4]
    beta = cal_vec[5]

    mx = (pixel[0] - cx) / fx
    my = (pixel[1] - cy) / fy
    r2 = mx * mx + my * my

    # Solve: 1 - (1-alpha)*k = alpha*sqrt(beta*r² + k²)
    # Square: (1-(1-alpha)*k)² = alpha²*(beta*r² + k²)
    # Expand: 1 - 2(1-a)*k + (1-a)²*k² = a²*beta*r² + a²*k²
    # ((1-a)² - a²)*k² - 2(1-a)*k + (1 - a²*beta*r²) = 0
    # (1 - 2*a)*k² - 2(1-a)*k + (1 - a²*beta*r²) = 0
    A = 1.0 - 2.0 * alpha
    B = -2.0 * (1.0 - alpha)
    C = 1.0 - alpha * alpha * beta * r2

    if abs(A) < 1e-12:
        # Linear case (alpha ≈ 0.5)
        if abs(B) < 1e-12:
            return None
        k = -C / B
    else:
        disc = B * B - 4.0 * A * C
        if disc < 0:
            return None
        sqrt_disc = np.sqrt(disc)
        # Pick the solution where k > 0 (point in front)
        k1 = (-B + sqrt_disc) / (2.0 * A)
        k2 = (-B - sqrt_disc) / (2.0 * A)
        # Also check that 1-(1-alpha)*k > 0 (needed for sqrt to be positive)
        valid = []
        for k_candidate in [k1, k2]:
            if k_candidate > 0 and (1.0 - (1.0 - alpha) * k_candidate) > 0:
                valid.append(k_candidate)
        if not valid:
            # Try the one closest to positive
            k = k1 if k1 > k2 else k2
        else:
            k = min(valid)  # pick smaller positive k (closest point)

    # ray = [mx, my, k] (unnormalized direction)
    ray = np.array([mx, my, k])
    return ray / np.linalg.norm(ray)


def unproject_ucm(pixel: np.ndarray, cal_vec: np.ndarray) -> np.ndarray | None:
    """UCM unprojection (alpha form) → unit ray [X, Y, Z], closed form.

    The double-sphere inverse at xi = 0 (DS with xi = 0 is UCM):
        mz = (1 - alpha² r²) / (alpha sqrt(1 - (2 alpha - 1) r²) + 1 - alpha)
    Rays behind the camera (Z < 0) are returned; pixels outside the image of
    the valid region (alpha > 0.5 and r² >= 1 / (2 alpha - 1)) give None.
    """
    fx, fy, cx, cy = cal_vec[0], cal_vec[1], cal_vec[2], cal_vec[3]
    alpha = cal_vec[4]

    mx = (pixel[0] - cx) / fx
    my = (pixel[1] - cy) / fy
    r2 = mx * mx + my * my

    disc = 1.0 - (2.0 * alpha - 1.0) * r2
    if disc <= 0:
        return None
    mz = (1.0 - alpha * alpha * r2) / (alpha * np.sqrt(disc) + 1.0 - alpha)
    ray = np.array([mx, my, mz])
    return ray / np.linalg.norm(ray)


# ---------------------------------------------------------------------------
# Omni-RadTan model (Mei 2007, a.k.a. MEI / unified omnidirectional)
# cal_vec: [fx, fy, cx, cy, xi, k1, k2, p1, p2]  (Kalibr omni-radtan)
# A legacy 8-long cal_vec (no p2) is read with p2 = 0.
# ---------------------------------------------------------------------------

def project_omni_radtan(
    p_cam: np.ndarray,
    cal_vec: np.ndarray,
    compute_jacobians: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Omni-RadTan (MEI) model projection.

    Forward:
        d = ||P||
        rz = 1 / (Z + xi * d)
        mx = X * rz,  my = Y * rz          (normalized image plane)
        r² = mx² + my²
        radial = 1 + k1*r² + k2*r⁴
        mx_d = mx * radial + 2*p1*mx*my + p2*(r² + 2*mx²)
        my_d = my * radial + p1*(r² + 2*my²) + 2*p2*mx*my
        u = fx * mx_d + cx,  v = fy * my_d + cy

    Validity: Z + xi * d > 0
    """
    X, Y, Z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
    fx, fy, cx, cy = cal_vec[0], cal_vec[1], cal_vec[2], cal_vec[3]
    xi = cal_vec[4]
    k1 = cal_vec[5]
    k2 = cal_vec[6]
    p1 = cal_vec[7]
    p2 = float(cal_vec[8]) if len(cal_vec) > 8 else 0.0

    d = np.sqrt(X * X + Y * Y + Z * Z)
    if d < 1e-10:
        return None, None, None

    denom = Z + xi * d
    if denom <= 1e-10:
        return None, None, None

    rz = 1.0 / denom
    mx = X * rz
    my = Y * rz

    # Radial-tangential distortion
    r2 = mx * mx + my * my
    r4 = r2 * r2
    radial = 1.0 + k1 * r2 + k2 * r4
    mx_d = mx * radial + 2.0 * p1 * mx * my + p2 * (r2 + 2.0 * mx * mx)
    my_d = my * radial + p1 * (r2 + 2.0 * my * my) + 2.0 * p2 * mx * my

    u = fx * mx_d + cx
    v = fy * my_d + cy
    pixel = np.array([u, v])

    if not compute_jacobians:
        return pixel, None, None

    # === Jacobians ===
    inv_d = 1.0 / d
    inv_denom = rz  # = 1/denom
    inv_denom2 = inv_denom * inv_denom

    # --- d(d)/d(X,Y,Z) ---
    dd_dX = X * inv_d
    dd_dY = Y * inv_d
    dd_dZ = Z * inv_d

    # --- d(rz)/d(X,Y,Z) --- rz = 1/(Z + xi*d), d(rz)/d(.) = -rz² * d(denom)/d(.)
    ddenom_dX = xi * dd_dX
    ddenom_dY = xi * dd_dY
    ddenom_dZ = 1.0 + xi * dd_dZ

    drz_dX = -inv_denom2 * ddenom_dX
    drz_dY = -inv_denom2 * ddenom_dY
    drz_dZ = -inv_denom2 * ddenom_dZ

    # --- d(mx)/d(X,Y,Z) --- mx = X * rz
    dmx_dX = rz + X * drz_dX
    dmx_dY = X * drz_dY
    dmx_dZ = X * drz_dZ

    dmy_dX = Y * drz_dX
    dmy_dY = rz + Y * drz_dY
    dmy_dZ = Y * drz_dZ

    # --- d(r2)/d(mx,my) --- r2 = mx² + my²
    dr2_dmx = 2.0 * mx
    dr2_dmy = 2.0 * my

    # --- d(radial)/d(mx,my) --- radial = 1 + k1*r2 + k2*r4
    drad_dr2 = k1 + 2.0 * k2 * r2
    drad_dmx = drad_dr2 * dr2_dmx
    drad_dmy = drad_dr2 * dr2_dmy

    # --- d(mx_d)/d(mx,my) --- mx_d = mx*radial + 2*p1*mx*my + p2*(r2 + 2*mx²)
    dmxd_dmx = radial + mx * drad_dmx + 2.0 * p1 * my + 6.0 * p2 * mx
    dmxd_dmy = mx * drad_dmy + 2.0 * p1 * mx + 2.0 * p2 * my

    # --- d(my_d)/d(mx,my) --- my_d = my*radial + p1*(r2 + 2*my²) + 2*p2*mx*my
    dmyd_dmx = my * drad_dmx + p1 * dr2_dmx + 2.0 * p2 * my
    dmyd_dmy = radial + my * drad_dmy + p1 * (dr2_dmy + 4.0 * my) + 2.0 * p2 * mx

    # --- Chain rule: d(pixel)/d(X,Y,Z) ---
    # du/dX = fx * (dmxd_dmx * dmx_dX + dmxd_dmy * dmy_dX)
    J_point = np.array([
        [fx * (dmxd_dmx * dmx_dX + dmxd_dmy * dmy_dX),
         fx * (dmxd_dmx * dmx_dY + dmxd_dmy * dmy_dY),
         fx * (dmxd_dmx * dmx_dZ + dmxd_dmy * dmy_dZ)],
        [fy * (dmyd_dmx * dmx_dX + dmyd_dmy * dmy_dX),
         fy * (dmyd_dmx * dmx_dY + dmyd_dmy * dmy_dY),
         fy * (dmyd_dmx * dmx_dZ + dmyd_dmy * dmy_dZ)],
    ])

    # --- Jacobian w.r.t. cal_vec [fx, fy, cx, cy, xi, k1, k2, p1, p2] ---
    # (2 x 8 for a legacy 8-long cal_vec, 2 x 9 with p2)
    J_cal = np.zeros((2, max(8, len(cal_vec))))
    J_cal[0, 0] = mx_d         # du/dfx
    J_cal[1, 1] = my_d         # dv/dfy
    J_cal[0, 2] = 1.0          # du/dcx
    J_cal[1, 3] = 1.0          # dv/dcy

    # d(pixel)/d(xi): rz depends on xi via denom = Z + xi*d
    # d(rz)/d(xi) = -rz² * d = -d * inv_denom²
    drz_dxi = -d * inv_denom2
    dmx_dxi = X * drz_dxi
    dmy_dxi = Y * drz_dxi
    J_cal[0, 4] = fx * (dmxd_dmx * dmx_dxi + dmxd_dmy * dmy_dxi)
    J_cal[1, 4] = fy * (dmyd_dmx * dmx_dxi + dmyd_dmy * dmy_dxi)

    # d(pixel)/d(k1): radial depends on k1 via k1*r2
    J_cal[0, 5] = fx * mx * r2         # du/dk1 = fx * mx * r2
    J_cal[1, 5] = fy * my * r2         # dv/dk1 = fy * my * r2

    # d(pixel)/d(k2): radial depends on k2 via k2*r4
    J_cal[0, 6] = fx * mx * r4         # du/dk2 = fx * mx * r4
    J_cal[1, 6] = fy * my * r4         # dv/dk2 = fy * my * r4

    # d(pixel)/d(p1): mx_d has 2*p1*mx*my, my_d has p1*(r2+2*my²)
    J_cal[0, 7] = fx * 2.0 * mx * my           # du/dp1
    J_cal[1, 7] = fy * (r2 + 2.0 * my * my)    # dv/dp1

    # d(pixel)/d(p2): mx_d has p2*(r2+2*mx²), my_d has 2*p2*mx*my
    if len(cal_vec) > 8:
        J_cal[0, 8] = fx * (r2 + 2.0 * mx * mx)    # du/dp2
        J_cal[1, 8] = fy * 2.0 * mx * my           # dv/dp2

    return pixel, J_point, J_cal


def _radtan_distort(mx, my, k1, k2, p1, p2):
    """Radtan distortion (k1, k2, p1, p2) of normalized (mx, my) and its 2x2
    Jacobian d(mx_d, my_d)/d(mx, my) as (J00, J01, J10, J11) — the Newton step
    of unproject_omni_radtan."""
    r2 = mx * mx + my * my
    r4 = r2 * r2
    radial = 1.0 + k1 * r2 + k2 * r4
    mx_d = mx * radial + 2.0 * p1 * mx * my + p2 * (r2 + 2.0 * mx * mx)
    my_d = my * radial + p1 * (r2 + 2.0 * my * my) + 2.0 * p2 * mx * my
    drad_dr2 = k1 + 2.0 * k2 * r2
    J00 = radial + mx * drad_dr2 * 2.0 * mx + 2.0 * p1 * my + 6.0 * p2 * mx
    J01 = mx * drad_dr2 * 2.0 * my + 2.0 * p1 * mx + 2.0 * p2 * my
    J10 = my * drad_dr2 * 2.0 * mx + p1 * 2.0 * mx + 2.0 * p2 * my
    J11 = (radial + my * drad_dr2 * 2.0 * my + p1 * (2.0 * my + 4.0 * my)
           + 2.0 * p2 * mx)
    return mx_d, my_d, (J00, J01, J10, J11)


def unproject_omni_radtan(
    pixel: np.ndarray,
    cal_vec: np.ndarray,
) -> np.ndarray | None:
    """Inverse projection for omni-radtan (MEI) model.

    Inverts pixel → ray direction in camera frame.
    Uses iterative undistortion (Newton-Raphson).
    """
    fx, fy, cx, cy = cal_vec[0], cal_vec[1], cal_vec[2], cal_vec[3]
    xi = cal_vec[4]
    k1 = cal_vec[5]
    k2 = cal_vec[6]
    p1 = cal_vec[7]
    p2 = float(cal_vec[8]) if len(cal_vec) > 8 else 0.0

    if abs(fx) < 1e-10 or abs(fy) < 1e-10:
        return None

    # Pixel → distorted normalized coordinates
    mx_d = (float(pixel[0]) - cx) / fx
    my_d = (float(pixel[1]) - cy) / fy

    # Iterative undistortion (Newton-Raphson)
    mx = mx_d
    my = my_d
    for _ in range(20):
        dist_x, dist_y, (J00, J01, J10, J11) = _radtan_distort(mx, my, k1, k2, p1, p2)
        dx = dist_x - mx_d
        dy = dist_y - my_d
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            break
        det = J00 * J11 - J01 * J10
        if abs(det) < 1e-15:
            break
        mx -= (J11 * dx - J01 * dy) / det
        my -= (-J10 * dx + J00 * dy) / det

    # Invert MEI projection: given undistorted (mx, my) and xi → unit ray
    # On the unit sphere: P = [mx*lam, my*lam, lam - xi]
    # where lam = (xi + sqrt(1 + (1 - xi²) * r²)) / (1 + r²)
    r2 = mx * mx + my * my
    disc = 1.0 + (1.0 - xi * xi) * r2
    if disc < 0:
        return None
    lam = (xi + np.sqrt(disc)) / (1.0 + r2)
    ray = np.array([mx * lam, my * lam, lam - xi])
    norm = np.linalg.norm(ray)
    if norm < 1e-10:
        return None
    return ray / norm


# ---------------------------------------------------------------------------
# Kannala-Brandt KB-N (Kannala & Brandt 2006), generic number of radial terms
# cal_vec: [fx, fy, cx, cy, k1, ..., kN]   (N = len(cal_vec) - 4)
# The registered model is kb4 (N = 4). It has the polynomial of
# gtsam Cal3Fisheye ('equidistant') with theta from atan2, so it matches gtsam for
# Z > 0 and stays valid past 90 deg.
# ---------------------------------------------------------------------------

_KB_AXIS_EPS = 1e-9        # rho < eps * |P| (Z > 0): on-axis series
_KB_NEWTON_ITERS = 100     # Newton, bisection-safeguarded


def _kb_poly(theta, ks):
    """(p, dp): theta_d = theta * p with p = 1 + sum k_i theta^(2i), and
    d(theta_d)/d(theta) = dp = 1 + sum (2i+1) k_i theta^(2i)."""
    th2 = theta * theta
    p = dp = t = 1.0
    for i, k in enumerate(ks, 1):
        t *= th2
        p += k * t
        dp += (2 * i + 1) * k * t
    return p, dp


@lru_cache(maxsize=64)
def _kb_fold(ks: tuple) -> float:
    """First fold of theta_d(theta) on (0, pi): the smallest theta > 0 with
    d(theta_d)/d(theta) = 0, else pi. dtd is a polynomial in u = theta², so
    its roots are exact; the monotonic (invertible) branch is [0, fold), and a
    polynomial that folds and rises again must not be valid on the later
    rising branch (its pixels alias first-branch pixels)."""
    if not any(ks):
        return math.pi
    # dtd(u) = 1 + sum (2i+1) k_i u^i, highest power first for np.roots
    coeffs = [(2 * i + 1) * k for i, k in enumerate(ks, 1)][::-1] + [1.0]
    fold = math.pi
    for z in np.roots(coeffs):
        u = float(z.real)
        if abs(z.imag) <= 1e-9 * max(1.0, abs(u)) and 0.0 < u < fold * fold:
            fold = math.sqrt(u)
    return fold


def project_kb(
    p_cam: np.ndarray,
    cal_vec: np.ndarray,
    compute_jacobians: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Kannala-Brandt KB-N projection with analytical Jacobians.

    Math:
        rho = sqrt(X² + Y²),  theta = atan2(rho, Z)
        theta_d = theta * (1 + k1*theta² + ... + kN*theta^(2N))
        s = theta_d / rho                    (-> 1/Z on the optical axis)
        u = fx * s * X + cx,  v = fy * s * Y + cy

    Validity: theta < pi (so rays up to, not including, straight back) and
    theta below the first fold of theta_d(theta) (`_kb_fold`), i.e. inside
    the polynomial's first monotonic branch; beyond it the inverse is
    ambiguous (a later rising branch aliases it), so the point is invalid.
    Matches gtsam Cal3Fisheye for N = 4 and Z > 0 (theta = atan(r) there).
    """
    X, Y, Z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
    fx, fy, cx, cy = (float(cal_vec[0]), float(cal_vec[1]),
                      float(cal_vec[2]), float(cal_vec[3]))
    ks = [float(k) for k in cal_vec[4:]]

    rho2 = X * X + Y * Y
    r2 = rho2 + Z * Z
    if r2 < 1e-20:
        return None, None, None
    rho = math.sqrt(rho2)
    theta = math.atan2(rho, Z)
    if theta >= math.pi:                     # straight back (rho == 0, Z < 0)
        return None, None, None
    if theta >= _kb_fold(tuple(ks)):         # past the first fold
        return None, None, None
    p, dtd = _kb_poly(theta, ks)
    if dtd <= 0.0:                           # (tangent fold, numerically)
        return None, None, None

    on_axis = Z > 0.0 and rho < _KB_AXIS_EPS * math.sqrt(r2)
    if on_axis:
        # theta / rho = atan(t) / (t Z), t = rho / Z  ->  (1 - t²/3) / Z
        a = (1.0 - rho2 / (3.0 * Z * Z)) / Z
    else:
        a = theta / rho
    s = a * p                                # theta_d / rho
    pixel = np.array([fx * s * X + cx, fy * s * Y + cy])

    if not compute_jacobians:
        return pixel, None, None

    # --- Jacobian w.r.t. [X, Y, Z] ---
    # ds/dZ   = dtd * dtheta/dZ / rho = -dtd / r²
    # g       = (ds/drho) / rho = (dtd * Z / r² - s) / rho²
    #           (on axis: the series limit (2*k1 - 2/3) / Z³)
    # du/dX   = fx * (s + X² g),  du/dY = fx * X Y g,  du/dZ = fx * X * ds/dZ
    dsdZ = -dtd / r2
    if on_axis:
        g = (2.0 * (ks[0] if ks else 0.0) - 2.0 / 3.0) / (Z * Z * Z)
    else:
        g = (dtd * Z / r2 - s) / rho2
    J_point = np.array([
        [fx * (s + X * X * g), fx * X * Y * g, fx * X * dsdZ],
        [fy * X * Y * g, fy * (s + Y * Y * g), fy * Y * dsdZ],
    ])

    # --- Jacobian w.r.t. cal_vec [fx, fy, cx, cy, k1..kN] ---
    # du/dk_i = fx * X * theta^(2i+1) / rho = fx * X * a * theta^(2i)
    J_cal = np.zeros((2, max(8, len(cal_vec))))
    J_cal[0, 0] = s * X        # du/dfx
    J_cal[1, 1] = s * Y        # dv/dfy
    J_cal[0, 2] = 1.0          # du/dcx
    J_cal[1, 3] = 1.0          # dv/dcy
    th2 = theta * theta
    t = a
    for i in range(len(ks)):
        t *= th2
        J_cal[0, 4 + i] = fx * X * t
        J_cal[1, 4 + i] = fy * Y * t

    return pixel, J_point, J_cal


def unproject_kb(pixel: np.ndarray, cal_vec: np.ndarray) -> np.ndarray | None:
    """KB-N unprojection → unit ray [X, Y, Z].

    Solves theta_d(theta) = r_d by a bisection-safeguarded 1-D Newton from
    theta = min(r_d, fold/2), bracketed on [0, fold) — the first monotonic
    branch `project_kb` accepts (`_kb_fold`, pi without a fold); the root must
    converge inside it, else None (e.g. r_d beyond the fold's theta_d).
    ray = [sin(theta) * mx / r_d, sin(theta) * my / r_d, cos(theta)].
    """
    fx, fy, cx, cy = (float(cal_vec[0]), float(cal_vec[1]),
                      float(cal_vec[2]), float(cal_vec[3]))
    ks = [float(k) for k in cal_vec[4:]]
    if abs(fx) < 1e-10 or abs(fy) < 1e-10:
        return None
    mx = (float(pixel[0]) - cx) / fx
    my = (float(pixel[1]) - cy) / fy
    rd = math.hypot(mx, my)
    if rd < 1e-12:                           # optical axis: theta ~ r_d
        ray = np.array([mx, my, 1.0])
        return ray / np.linalg.norm(ray)

    # Safeguarded Newton: [lo, hi] brackets the root of the first monotonic
    # branch (f(0) = -r_d < 0, hi = the first fold); an iterate with dtd <= 0
    # or outside the bracket falls back to bisection, so Newton never settles
    # on a folded or later branch.
    fold = _kb_fold(tuple(ks))
    lo, hi = 0.0, fold
    theta = min(rd, 0.5 * fold)
    for _ in range(_KB_NEWTON_ITERS):
        p, dtd = _kb_poly(theta, ks)
        f = theta * p - rd
        if f == 0.0 and dtd > 0.0:           # exact root: done (no bisection)
            break
        if dtd <= 0.0 or f > 0.0:
            hi = theta
        else:
            lo = theta
        if dtd <= 0.0:
            step = theta - 0.5 * (lo + hi)
        else:
            step = f / dtd
            if not lo < theta - step < hi:
                step = theta - 0.5 * (lo + hi)
        theta -= step
        if abs(step) < 1e-15 * max(1.0, theta):
            break
    else:
        return None
    p, dtd = _kb_poly(theta, ks)
    if (theta >= fold or dtd <= 0.0
            or abs(theta * p - rd) > 1e-9 * max(1.0, rd)):
        return None
    st = math.sin(theta)
    return np.array([st * mx / rd, st * my / rd, math.cos(theta)])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CUSTOM_MODELS = CUSTOM     # registry: every non-gtsam model
