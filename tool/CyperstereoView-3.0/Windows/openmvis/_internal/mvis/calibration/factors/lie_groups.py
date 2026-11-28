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
"""Lie group Jacobian utilities for SO(3) and SE(3).

Implements the closed-form Jacobians needed for the analytical unified
projection factor:

- SO(3): right Jacobian J_r, left Jacobian inverse J_l⁻¹
- SE(3): right Jacobian J_r, left Jacobian inverse J_l⁻¹, Adjoint
- Skew-symmetric matrix [v]×

References:
- Barfoot, "State Estimation for Robotics" (2017), Ch. 7 & 9
- Solà, "A micro Lie theory for state estimation in robotics" (2018)
"""

import numpy as np
import gtsam

# Small-angle threshold for Taylor expansion
_EPS = 1e-10


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix [v]× from a 3-vector.

    [v]× · w = v × w (cross product).
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def exp_so3(phi: np.ndarray) -> np.ndarray:
    """SO(3) exponential: rotation matrix from a 3-vector (gtsam Rot3.Expmap)."""
    return gtsam.Rot3.Expmap(np.asarray(phi, float)).matrix()


def log_so3(R: np.ndarray) -> np.ndarray:
    """SO(3) logarithm: 3-vector from a rotation matrix (gtsam Rot3.Logmap)."""
    return gtsam.Rot3.Logmap(gtsam.Rot3(np.asarray(R, float)))


def vee_so3(W: np.ndarray) -> np.ndarray:
    """Vee operator: skew-symmetric 3×3 matrix → 3-vector (inverse of skew)."""
    return np.array([W[2, 1], W[0, 2], W[1, 0]])


# =====================================================================
# SE(3) exp / log / hat / inverse
#
# Tangent 6-vectors use gtsam ordering xi = [omega(3), rho(3)] (rotation
# first), consistent with gtsam.Pose3.Expmap/Logmap. Poses are 4×4
# homogeneous matrices.
# =====================================================================

def hat_se3(xi: np.ndarray) -> np.ndarray:
    """se(3) 4×4 matrix from a 6-vector [omega(3), rho(3)]."""
    M = np.zeros((4, 4))
    M[0:3, 0:3] = skew(xi[0:3])
    M[0:3, 3] = xi[3:6]
    return M


def exp_se3(xi: np.ndarray) -> np.ndarray:
    """SE(3) exponential: 4×4 pose from a 6-vector (gtsam Pose3.Expmap)."""
    return gtsam.Pose3.Expmap(np.asarray(xi, float)).matrix()


def log_se3(T: np.ndarray) -> np.ndarray:
    """SE(3) logarithm: 6-vector from a 4×4 pose (gtsam Pose3.Logmap)."""
    return gtsam.Pose3.Logmap(gtsam.Pose3(np.asarray(T, float)))


def inv_se3(T: np.ndarray) -> np.ndarray:
    """Inverse of a 4×4 SE(3) homogeneous matrix (closed form)."""
    T = np.asarray(T, float)
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    out = np.eye(4)
    out[0:3, 0:3] = R.T
    out[0:3, 3] = -R.T @ p
    return out


# =====================================================================
# SO(3) Jacobians
# =====================================================================

def so3_right_jacobian(phi: np.ndarray) -> np.ndarray:
    """Right Jacobian of SO(3): J_r(φ).

    J_r(φ) = I - (1-cos‖φ‖)/‖φ‖² · [φ]× + (‖φ‖-sin‖φ‖)/‖φ‖³ · [φ]×²

    For small ‖φ‖: J_r ≈ I - 0.5·[φ]×

    Args:
        phi: 3-vector (rotation axis × angle).

    Returns:
        3×3 matrix.
    """
    theta = np.linalg.norm(phi)
    if theta < _EPS:
        return np.eye(3) - 0.5 * skew(phi)

    phi_x = skew(phi)
    phi_x2 = phi_x @ phi_x
    return (np.eye(3)
            - (1.0 - np.cos(theta)) / (theta ** 2) * phi_x
            + (theta - np.sin(theta)) / (theta ** 3) * phi_x2)


def so3_left_jacobian(phi: np.ndarray) -> np.ndarray:
    """Left Jacobian of SO(3): J_l(φ).

    J_l(φ) = I + (1-cos‖φ‖)/‖φ‖² · [φ]× + (‖φ‖-sin‖φ‖)/‖φ‖³ · [φ]×²

    Note: J_l(φ) = J_r(-φ).
    """
    return so3_right_jacobian(-phi)


def so3_left_jacobian_inv(phi: np.ndarray) -> np.ndarray:
    """Inverse of the left Jacobian of SO(3): J_l⁻¹(φ).

    J_l⁻¹(φ) = I - 0.5·[φ]× + (1/‖φ‖² - (1+cos‖φ‖)/(2‖φ‖sin‖φ‖)) · [φ]×²

    For small ‖φ‖: J_l⁻¹ ≈ I + 0.5·[φ]×
    """
    theta = np.linalg.norm(phi)
    if theta < _EPS:
        return np.eye(3) + 0.5 * skew(phi)

    phi_x = skew(phi)
    phi_x2 = phi_x @ phi_x
    c = 1.0 / (theta ** 2) - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
    return np.eye(3) - 0.5 * phi_x + c * phi_x2


def so3_right_jacobian_inv(phi: np.ndarray) -> np.ndarray:
    """Inverse of the right Jacobian of SO(3): J_r⁻¹(φ).

    J_r⁻¹(φ) = J_l⁻¹(-φ).
    """
    return so3_left_jacobian_inv(-phi)


# =====================================================================
# SE(3) Jacobians
# =====================================================================

def _Q_matrix(xi: np.ndarray) -> np.ndarray:
    """Q matrix for the SE(3) Jacobian (Barfoot eq. 9.84).

    Used in the upper-right block of the SE(3) right/left Jacobian.

    Args:
        xi: 6-vector [rho; phi] where rho is translation, phi is rotation.

    Returns:
        3×3 matrix.
    """
    rho = xi[:3]
    phi = xi[3:]
    theta = np.linalg.norm(phi)

    rho_x = skew(rho)
    phi_x = skew(phi)

    if theta < _EPS:
        return 0.5 * rho_x

    theta2 = theta ** 2
    theta3 = theta ** 3
    theta4 = theta ** 4
    theta5 = theta ** 5
    st = np.sin(theta)
    ct = np.cos(theta)

    # Barfoot eq. 9.84
    Q = (0.5 * rho_x
         + (theta - st) / theta3 * (phi_x @ rho_x + rho_x @ phi_x + phi_x @ rho_x @ phi_x)
         - (1.0 - theta2 / 2.0 - ct) / theta4 * (phi_x @ phi_x @ rho_x + rho_x @ phi_x @ phi_x
                                                    - 3.0 * phi_x @ rho_x @ phi_x)
         - 0.5 * ((1.0 - theta2 / 2.0 - ct) / theta4
                   - 3.0 * (theta - st - theta3 / 6.0) / theta5) * (
             phi_x @ rho_x @ phi_x @ phi_x + phi_x @ phi_x @ rho_x @ phi_x))

    return Q


def se3_right_jacobian(xi: np.ndarray) -> np.ndarray:
    """Right Jacobian of SE(3): J_r(ξ).

    GTSAM convention: ξ = [ω₁,ω₂,ω₃, v₁,v₂,v₃] (rotation first).

    In GTSAM convention, J_r has the structure:
        J_r = [J_r(φ),     0₃ₓ₃  ]
              [Q_gtsam,    J_r(φ) ]

    where Q_gtsam is computed from the Barfoot Q with convention mapping.

    Args:
        xi: 6-vector in GTSAM convention [rot; trans].

    Returns:
        6×6 matrix in GTSAM convention.
    """
    phi = xi[:3]   # rotation part (GTSAM: first 3)
    rho = xi[3:]   # translation part (GTSAM: last 3)

    Jr_so3 = so3_right_jacobian(phi)

    # Q matrix from Barfoot uses [trans; rot] ordering internally
    xi_barfoot = np.array([rho[0], rho[1], rho[2], phi[0], phi[1], phi[2]])
    Qr = _Q_matrix(xi_barfoot)

    # Build directly in GTSAM convention [rot; trans]
    J = np.zeros((6, 6))
    J[:3, :3] = Jr_so3      # rot-rot block
    J[3:, 3:] = Jr_so3      # trans-trans block
    J[3:, :3] = Qr.T        # trans-rot coupling (transpose for convention)
    return J


def se3_left_jacobian_inv(xi: np.ndarray) -> np.ndarray:
    """Inverse left Jacobian of SE(3): J_l⁻¹(ξ).

    Args:
        xi: 6-vector in GTSAM convention [rot; trans].

    Returns:
        6×6 matrix in GTSAM convention.
    """
    phi = xi[:3]
    rho = xi[3:]

    Jl_inv_so3 = so3_left_jacobian_inv(phi)

    # Q for left Jacobian inverse: uses -xi in Barfoot convention
    xi_barfoot_neg = np.array([-rho[0], -rho[1], -rho[2],
                                -phi[0], -phi[1], -phi[2]])
    Ql = _Q_matrix(xi_barfoot_neg)

    # The Barfoot upper-right block is -J_l_inv · Q_l · J_l_inv
    coupling = -Jl_inv_so3 @ Ql @ Jl_inv_so3

    # Build directly in GTSAM convention [rot; trans]
    J = np.zeros((6, 6))
    J[:3, :3] = Jl_inv_so3          # rot-rot block
    J[3:, 3:] = Jl_inv_so3          # trans-trans block
    J[3:, :3] = coupling.T          # trans-rot coupling (transpose)
    return J


def se3_adjoint(T: gtsam.Pose3) -> np.ndarray:
    """Adjoint representation of SE(3): Adj(T).

    In Barfoot convention (trans first):
        Adj(T) = [R,     [t]× · R]
                 [0₃ₓ₃,     R    ]

    In GTSAM convention (rot first):
        Adj(T) = [R,        0₃ₓ₃ ]
                 [[t]×·R,     R   ]

    Args:
        T: gtsam.Pose3.

    Returns:
        6×6 matrix in GTSAM convention [rot; trans].
    """
    R = T.rotation().matrix()
    t = T.translation()
    t_x = skew(np.array([t[0], t[1], t[2]]))

    # GTSAM convention: [rot; trans]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R           # rot-rot block
    Ad[3:, :3] = t_x @ R    # trans-rot block
    Ad[3:, 3:] = R           # trans-trans block
    return Ad

