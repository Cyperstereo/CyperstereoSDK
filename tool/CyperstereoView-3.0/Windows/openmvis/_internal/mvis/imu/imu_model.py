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
"""Abstract IMU intrinsic-model interface (Kalibr / RPNG / Calyx).

This is the single home for per-model IMU-intrinsics math: the correction
matrices (Dw / Da / Tg), their parameter Jacobians, the 24-dim manifold ops, and
the forward/inverse measurement model (`measure` / `correct`). Both the cam-IMU
factors/integrator and the simulator consume it polymorphically — the per-model
layout (triangular vs full, presence of a free rotation DOF, block offsets) lives
in the three subclasses (`KalibrImu`, `RpngImu`, `CalyxImu`), not in
`if model == ...` branches scattered across the pipeline.

Ported from MVIS `ImuIntrinsics.h`; validated against the C++ `ImuIntrinsics` /
`ACI_In` oracle. The model variants:

    KALIBR (0): Dw, Da lower-triangular (6); free rotation R_wtoI (R_atoI = I)
    RPNG   (1): Dw, Da upper-triangular (6); free rotation R_atoI (R_wtoI = I)
    CALYX  (2): Dw full 3x3 (9), Da upper-tri (6); NO free rotation — the
                gyro->IMU rotation is absorbed into the full Dw and recovered
                post-hoc by RQ decomposition (`decompose`).

All three total 24 manifold dims:
    KALIBR/RPNG: [d_Dw(6), d_Da(6), d_Tg(9), d_R(3)]
    CALYX:       [d_Dw(9), d_Da(6), d_Tg(9)]          (no rotation DOF)

Python-port note: GTSAM-Python `Values` cannot hold a custom 24-dim manifold, so
in the factor graph this state is DECOMPOSED into native GTSAM types: Rot3 (R) +
Vector (x_Dw ‖ x_Da ‖ x_Tg). This class is the value-side representation + matrix
builders; the factor reads the decomposed keys and rebuilds the matrices here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import gtsam

KALIBR = 0
RPNG = 1
CALYX = 2

# Identity defaults — the 6-vec that builds the 3x3 IDENTITY depends on the
# triangular convention (the diagonal lives at different vector positions):
#   upper-tri (RPNG / Calyx-Da): diag at [0,2,5] -> [1,0,1,0,0,1]
#   lower-tri (Kalibr):          diag at [0,3,5] -> [1,0,0,1,0,1]
# Using the wrong one builds a rank-deficient seed (the original Kalibr bug).
_ID_UPPER = np.array([1, 0, 1, 0, 0, 1], dtype=float)
_ID_LOWER = np.array([1, 0, 0, 1, 0, 1], dtype=float)
_DEFAULT_DW = _ID_UPPER                                       # back-compat alias
_DEFAULT_DA = _ID_UPPER
_DEFAULT_DW_FULL = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=float)  # full 3x3
_DEFAULT_TG = np.zeros(9, dtype=float)

_I3 = np.eye(3)
_I3.setflags(write=False)   # shared by the R_wtoI / R_atoI identity properties
_E = np.eye(3)  # columns e1, e2, e3

# IMU-intrinsics staging strings (which blocks are freed in the solve).
IMU_STAGES = ("pinned", "dw_da", "dw_da_r", "full")

_MODEL_NAMES = {KALIBR: "KALIBR", RPNG: "RPNG", CALYX: "CALYX"}
_NAME_TO_MODEL = {"kalibr": KALIBR, "rpng": RPNG, "calyx": CALYX}


# --- shared correction-matrix builders (the per-model math, in one place) ---
def lower_tri(x: np.ndarray) -> np.ndarray:
    """3x3 lower-triangular matrix from a 6-vector (Kalibr Dw/Da convention)."""
    return np.array([
        [x[0], 0.0,  0.0],
        [x[1], x[3], 0.0],
        [x[2], x[4], x[5]],
    ])


def upper_tri(x: np.ndarray) -> np.ndarray:
    """3x3 upper-triangular matrix from a 6-vector (RPNG / Calyx-Da convention)."""
    return np.array([
        [x[0], x[1], x[3]],
        [0.0,  x[2], x[4]],
        [0.0,  0.0,  x[5]],
    ])


def full3x3(x: np.ndarray) -> np.ndarray:
    """Full 3x3 matrix from a 9-vector, row-major (Calyx Dw convention)."""
    return np.array([
        [x[0], x[1], x[2]],
        [x[3], x[4], x[5]],
        [x[6], x[7], x[8]],
    ])


def tg_matrix(x_tg: np.ndarray) -> np.ndarray:
    """Build the 3x3 Tg (gyro g-sensitivity) matrix, column-major from a 9-vector.
    Model-independent (Tg has the same parameterization for all models)."""
    return np.array([
        [x_tg[0], x_tg[3], x_tg[6]],
        [x_tg[1], x_tg[4], x_tg[7]],
        [x_tg[2], x_tg[5], x_tg[8]],
    ])


# --- shared parameter Jacobians of the correction matrices (used by the ACI
#     integrator); `lower`/`upper`/`full` mirror the triangular conventions ---
def jac_lower(v: np.ndarray) -> np.ndarray:
    """d(M·v)/d(x) for a lower-triangular 3x3 M(x6): 3x6 (Kalibr Dw/Da)."""
    v1, v2, v3 = v
    H = np.zeros((3, 6))
    H[:, 0:3] = v1 * _I3
    H[:, 3] = v2 * _E[:, 1]
    H[:, 4] = v2 * _E[:, 2]
    H[:, 5] = v3 * _E[:, 2]
    return H


def jac_upper(v: np.ndarray) -> np.ndarray:
    """d(M·v)/d(x) for an upper-triangular 3x3 M(x6): 3x6 (RPNG / Calyx-Da)."""
    v1, v2, v3 = v
    H = np.zeros((3, 6))
    H[:, 0] = v1 * _E[:, 0]
    H[:, 1] = v2 * _E[:, 0]
    H[:, 2] = v2 * _E[:, 1]
    H[:, 3:6] = v3 * _I3
    return H


def jac_full(v: np.ndarray) -> np.ndarray:
    """d(M·v)/d(x) for a full row-major 3x3 M(x9): 3x9 (Calyx Dw). Row i depends
    only on x[3i:3i+3] with d(row_i)/d(x[3i+j]) = v[j] -> block-diagonal of vᵀ."""
    H = np.zeros((3, 9))
    for i in range(3):
        H[i, 3 * i:3 * i + 3] = v
    return H


def tg_jacobian(a_inI: np.ndarray) -> np.ndarray:
    """d(Tg·a_inI)/d(x_Tg): 3x9 (matches ACI_In::get_H_Tg). Model-independent."""
    a1, a2, a3 = a_inI
    H = np.zeros((3, 9))
    H[:, 0:3] = a1 * _I3
    H[:, 3:6] = a2 * _I3
    H[:, 6:9] = a3 * _I3
    return H


def decompose_dw(Dw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RQ-style split of a full gyro correction: Dw = R_wtoI · U_w.

    R_wtoI is a proper rotation (gyro->IMU), U_w is upper-triangular with a
    POSITIVE diagonal (the gauge-fixed gyro scale/misalignment). Computed via QR
    (Dw = Q·U) with the column signs folded so diag(U) > 0; the resulting Q is a
    proper rotation when det(Dw) > 0 (the physical case, Dw ≈ I).
    """
    Dw = np.asarray(Dw, float)
    Q, U = np.linalg.qr(Dw)
    s = np.sign(np.diag(U))
    s[s == 0] = 1.0
    D = np.diag(s)
    Q = Q @ D
    U = D @ U                      # diag(U) now > 0; Q·U == Dw unchanged
    return Q, U


# --- model resolution helpers ---
def parse_imu_model(name) -> int:
    """Resolve an IMU-model name ('kalibr'|'rpng'|'calyx') or int to its id."""
    if isinstance(name, (int, np.integer)):
        if int(name) in _MODEL_NAMES:
            return int(name)
        raise ValueError(f"unknown imu model id {name!r}")
    key = str(name).strip().lower()
    if key not in _NAME_TO_MODEL:
        raise ValueError(
            f"unknown imu model {name!r}; choose from {sorted(_NAME_TO_MODEL)}")
    return _NAME_TO_MODEL[key]


def imu_model_name(model: int) -> str:
    """Lower-case canonical name for a model id."""
    return _MODEL_NAMES[parse_imu_model(model)].lower()


# --- model-dispatched layout functions (forward to the concrete subclass; kept
#     as module functions for the hot factor path and back-compat callers) ---
def _model_class(model: int):
    classes = _model_classes()
    if isinstance(model, (int, np.integer)):        # hot path: a known model id
        cls = classes.get(int(model))
        if cls is not None:
            return cls
    return classes[parse_imu_model(model)]


def n_dw_for(model: int) -> int:
    """Number of gyro-correction params for a model (Calyx full 3x3 = 9, else 6)."""
    return _model_class(model).N_DW


def dw_matrix(x_dw: np.ndarray, model: int) -> np.ndarray:
    """Build the 3x3 Dw (gyro scale/misalignment) matrix from its param vector."""
    return _model_class(model).build_dw(x_dw)


def da_matrix(x_da: np.ndarray, model: int) -> np.ndarray:
    """Build the 3x3 Da (accel scale/misalignment) matrix from its 6-vector."""
    return _model_class(model).build_da(x_da)


def dw_jacobian(w_inw: np.ndarray, model: int) -> np.ndarray:
    """d(Dw·w_inw)/d(x_Dw): 3x6 (Kalibr/RPNG) or 3x9 (Calyx)."""
    return _model_class(model).dw_param_jacobian(w_inw)


def da_jacobian(a_ina: np.ndarray, model: int) -> np.ndarray:
    """d(Da·a_ina)/d(x_Da): 3x6 (matches ACI_In::get_H_Da)."""
    return _model_class(model).da_param_jacobian(a_ina)


def intrinsic_block_slices(model: int) -> dict:
    """Map block name -> slice within the 24-dim intrinsic tangent, per model.

    Kalibr/RPNG: dw[0:6] da[6:12] tg[12:21] r[21:24]
    Calyx:       dw[0:9] da[9:15] tg[15:24]   (no rotation block)
    """
    n = n_dw_for(model)
    sl = {"dw": slice(0, n), "da": slice(n, n + 6), "tg": slice(n + 6, n + 15)}
    if model != CALYX:
        sl["r"] = slice(n + 15, n + 18)
    return sl


def freed_intrinsic_blocks(model: int, stage: str) -> list:
    """Which intrinsic blocks a staging string frees, for this model.

    Calyx has no separate 'r' block (the gyro->IMU rotation lives inside the full
    Dw), so 'dw_da_r' frees the same blocks as 'dw_da' for Calyx.
    """
    if stage not in IMU_STAGES:
        raise ValueError(f"unknown imu_intrinsics stage {stage!r}")
    if stage == "pinned":
        return []
    blocks = ["dw", "da"]
    if stage in ("dw_da_r", "full") and model != CALYX:
        blocks.append("r")
    if stage == "full":
        blocks.append("tg")
    return blocks


def intrinsic_sigma_vector(model: int, stage: str, *, sigma_dw_da: float,
                           sigma_tg: float, sigma_R: float, tiny: float,
                           freed=None) -> np.ndarray:
    """Length-24 prior-sigma vector for the intrinsic tangent.

    Freed blocks get their weak prior; everything else gets `tiny`. `freed`
    overrides the stage->blocks policy (e.g. the gyro-only aux frees only ['dw']).
    """
    v = np.full(24, tiny)
    sl = intrinsic_block_slices(model)
    blocks = freed if freed is not None else freed_intrinsic_blocks(model, stage)
    sig = {"dw": sigma_dw_da, "da": sigma_dw_da, "tg": sigma_tg, "r": sigma_R}
    for b in blocks:
        if b in sl:
            v[sl[b]] = sig[b]
    return v


class ImuModel(ABC):
    """Abstract base for an IMU intrinsic model (24-dim manifold value).

    Carries the value-side parameters (x_dw, x_da, x_tg, R) for ONE calibration;
    the concrete subclass fixes the layout (triangular/full, free-rotation DOF).
    Construct a specific variant directly (`KalibrImu(...)`), via the factory
    (`create_imu_model(model, ...)`), or via the base with a `model=` kwarg
    (`ImuModel(model=KALIBR)`), which dispatches to the right subclass.

    Subclasses provide the static builders (`build_dw`, `build_da`,
    `dw_param_jacobian`, `da_param_jacobian`, `identity_dw`, `identity_da`) and
    the class attributes MODEL / NAME / N_DW / HAS_R; everything else (matrices,
    manifold ops, the measurement model) is shared here.
    """

    DIM = 24

    # Subclass-provided identity:
    MODEL: int = None
    NAME: str = None
    N_DW: int = None
    HAS_R: bool = None

    # --- construction (base dispatches to the subclass when `model=` is given) ---
    def __new__(cls, x_dw=None, x_da=None, x_tg=None, R=None, model=None):
        if cls is ImuModel:
            m = RPNG if model is None else parse_imu_model(model)
            cls = _model_classes()[m]
        return object.__new__(cls)

    def __init__(self, x_dw=None, x_da=None, x_tg=None, R=None, model=None):
        if model is not None and parse_imu_model(model) != self.MODEL:
            raise ValueError(
                f"model={model!r} conflicts with {type(self).__name__} "
                f"({imu_model_name(self.MODEL)})")
        self.model = self.MODEL                       # back-compat int attribute
        self.x_dw = self.identity_dw() if x_dw is None else np.asarray(x_dw, float)
        self.x_da = self.identity_da() if x_da is None else np.asarray(x_da, float)
        self.x_tg = _DEFAULT_TG.copy() if x_tg is None else np.asarray(x_tg, float)
        self.R = np.eye(3) if R is None else np.asarray(R, float)

    @classmethod
    def from_tangent(cls, xi, model: int = RPNG) -> "ImuModel":
        """Reconstruct absolute intrinsics from a 24-dim tangent ξ around identity.

        Inverse of how the pipeline stores the result (`imu_intr_xi` is ξ around
        the identity base used in preintegration), so callers should use this —
        never hand-slice ξ — to get the true Dw/Da/Tg for any model.
        """
        return create_imu_model(model).retract(np.asarray(xi, float))

    # --- per-model static math (implemented by the subclasses) ---
    @staticmethod
    @abstractmethod
    def build_dw(x_dw: np.ndarray) -> np.ndarray:
        """3x3 gyro correction Dw from its param vector (layout per model)."""

    @staticmethod
    @abstractmethod
    def build_da(x_da: np.ndarray) -> np.ndarray:
        """3x3 accel correction Da from its 6-vector (layout per model)."""

    @staticmethod
    @abstractmethod
    def dw_param_jacobian(w_inw: np.ndarray) -> np.ndarray:
        """d(Dw·w_inw)/d(x_Dw) (3x6 or 3x9)."""

    @staticmethod
    @abstractmethod
    def da_param_jacobian(a_ina: np.ndarray) -> np.ndarray:
        """d(Da·a_ina)/d(x_Da) (3x6)."""

    @staticmethod
    @abstractmethod
    def identity_dw() -> np.ndarray:
        """Param vector that builds the IDENTITY Dw for this model (fresh copy)."""

    @staticmethod
    @abstractmethod
    def identity_da() -> np.ndarray:
        """Param vector that builds the IDENTITY Da for this model (fresh copy)."""

    # --- correction matrices (consumed by the ACI factor) ---
    def Dw(self) -> np.ndarray:
        return type(self).build_dw(self.x_dw)

    def Da(self) -> np.ndarray:
        return type(self).build_da(self.x_da)

    def Tg(self) -> np.ndarray:
        return tg_matrix(self.x_tg)

    # The model's own rotation is returned BY REFERENCE (`self.R`, writeable)
    # where the model carries it; the fixed identity side is the shared
    # read-only `_I3` instead of a fresh `np.eye(3)` per call.
    @property
    def R_wtoI(self) -> np.ndarray:
        return self.R if self.MODEL == KALIBR else _I3

    @property
    def R_atoI(self) -> np.ndarray:
        return self.R if self.MODEL == RPNG else _I3

    def decompose(self) -> tuple[np.ndarray, np.ndarray]:
        """Calyx post-calibration split: full Dw -> (R_wtoI, U_w upper-tri).
        For Kalibr/RPNG (already factored) returns (R_wtoI, Dw)."""
        if self.MODEL == CALYX:
            return decompose_dw(self.Dw())
        return self.R_wtoI, self.Dw()

    # --- forward / inverse measurement model ---
    # Mirrors `simulation/imu_measurements.py` and the ACI correction in
    # `factors/aci.py`; the two are exact inverses in the noise-free case.
    def measure(self, w_inI, a_inI, b_g=None, b_a=None):
        """True body-frame (w_inI, a_inI) -> RAW IMU readings (w_m, a_m).

        Adds this model's intrinsics + biases to clean true values to mimic what
        a real IMU would output:
            a_m = Da^{-1} R_atoI^T a_inI + b_a
            w_m = Dw^{-1} R_wtoI^T w_inI + Tg a_inI + b_g
        (white/walk noise is the simulator's job — see `simulate_imu`.)
        """
        w_inI = np.asarray(w_inI, float)
        a_inI = np.asarray(a_inI, float)
        b_g = np.zeros(3) if b_g is None else np.asarray(b_g, float)
        b_a = np.zeros(3) if b_a is None else np.asarray(b_a, float)
        a_m = np.linalg.solve(self.Da(), self.R_atoI.T @ a_inI) + b_a
        w_inw = np.linalg.solve(self.Dw(), self.R_wtoI.T @ w_inI)
        w_m = w_inw + b_g + self.Tg() @ a_inI
        return w_m, a_m

    def correct(self, w_m, a_m, b_g=None, b_a=None):
        """RAW IMU readings (w_m, a_m) -> true body-frame (w_inI, a_inI).

        Removes this model's intrinsics + biases (the ACI correction); the exact
        inverse of `measure`:
            a_ina = a_m - b_a;   a_inI = R_atoI · Da · a_ina
            w_inw = w_m - b_g - Tg · a_inI;   w_inI = R_wtoI · Dw · w_inw
        """
        w_m = np.asarray(w_m, float)
        a_m = np.asarray(a_m, float)
        b_g = np.zeros(3) if b_g is None else np.asarray(b_g, float)
        b_a = np.zeros(3) if b_a is None else np.asarray(b_a, float)
        a_ina = a_m - b_a
        a_inI = self.R_atoI @ self.Da() @ a_ina
        w_inw = w_m - b_g - self.Tg() @ a_inI
        w_inI = self.R_wtoI @ self.Dw() @ w_inw
        return w_inI, a_inI

    # --- model-dependent tangent layout ---
    @property
    def n_dw(self) -> int:
        return self.N_DW

    @property
    def has_R(self) -> bool:
        """Whether the model carries a free SO(3) rotation DOF (Calyx does not)."""
        return self.HAS_R

    @property
    def da_off(self) -> int:
        return self.N_DW

    @property
    def tg_off(self) -> int:
        return self.N_DW + 6

    @property
    def r_off(self) -> int:
        return self.N_DW + 15

    # --- manifold ops (24-dim, layout per model) ---
    # Mirrors C++ ImuIntrinsics::retract/localCoordinates. The vector params
    # retract additively; for Kalibr/RPNG R retracts on SO(3) via R·Exp(d_R).
    def retract(self, xi: np.ndarray) -> "ImuModel":
        xi = np.asarray(xi, float)
        n = self.N_DW
        R = (self.R @ gtsam.Rot3.Expmap(xi[n + 15:n + 18]).matrix()
             if self.HAS_R else self.R)
        return type(self)(
            x_dw=self.x_dw + xi[0:n],
            x_da=self.x_da + xi[n:n + 6],
            x_tg=self.x_tg + xi[n + 6:n + 15],
            R=R)

    def local_coordinates(self, other: "ImuModel") -> np.ndarray:
        parts = [other.x_dw - self.x_dw, other.x_da - self.x_da,
                 other.x_tg - self.x_tg]
        if self.HAS_R:
            parts.append(gtsam.Rot3.Logmap(gtsam.Rot3(self.R.T @ other.R)))
        return np.concatenate(parts)

    @property
    def x_vec(self) -> np.ndarray:
        """[x_dw, x_da, x_tg] — the vector-space part (21 dims, or 24 for Calyx)."""
        return np.concatenate([self.x_dw, self.x_da, self.x_tg])

    # Back-compat alias (misnamed for Calyx, where it is 24-long).
    x21 = x_vec

    def copy(self) -> "ImuModel":
        return type(self)(self.x_dw.copy(), self.x_da.copy(), self.x_tg.copy(),
                          self.R.copy())

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(model={_MODEL_NAMES[self.MODEL]}, "
                f"x_dw={self.x_dw}, x_da={self.x_da}, x_tg={self.x_tg})")


_MODEL_CLASSES: dict | None = None


def _model_classes() -> dict:
    """Registry {model_id: subclass}; imported lazily to avoid an import cycle
    (the subclasses import this module) and cached at module level after the
    first call (`dw_jacobian` / `da_jacobian` sit on the per-step path).
    Mirrors `create_camera_model`'s style."""
    global _MODEL_CLASSES
    if _MODEL_CLASSES is None:
        from mvis.imu.kalibr import KalibrImu
        from mvis.imu.rpng import RpngImu
        from mvis.imu.calyx import CalyxImu
        _MODEL_CLASSES = {KALIBR: KalibrImu, RPNG: RpngImu, CALYX: CalyxImu}
    return _MODEL_CLASSES


def create_imu_model(model: int = RPNG, x_dw=None, x_da=None, x_tg=None,
                     R=None) -> ImuModel:
    """Factory: build the IMU model for `model` ('kalibr'|'rpng'|'calyx' or int).

    With no parameter vectors it returns the identity intrinsics for that model.
    """
    cls = _model_classes()[parse_imu_model(model)]
    return cls(x_dw=x_dw, x_da=x_da, x_tg=x_tg, R=R)
