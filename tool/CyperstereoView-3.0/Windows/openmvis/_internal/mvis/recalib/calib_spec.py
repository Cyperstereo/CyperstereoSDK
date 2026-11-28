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
"""calib_spec — structured calibration objects loaded from a MVIS result YAML.

Exports
-------
RigCalib   : top-level container (base_imu + aux_imus + cams)
ImuCalib   : per-IMU calibration (model, intrinsics, biases, td)
MVIS   : per-camera calibration (K, dist, td, tr)
load_rig_calib(path) -> RigCalib
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging

import numpy as np
import yaml

from mvis.core.camera_models import is_removed_model, removed_model_message
from mvis.imu import ImuModel

log = logging.getLogger("mvis.recalib.calib_spec")


@dataclass
class ImuCalib:
    """Calibration for a single IMU."""
    name: str
    model: str                          # e.g. "rpng"
    intr: ImuModel                      # full intrinsic model (reconstructed from xi)
    b_g: np.ndarray                     # gyro bias  (3,)
    b_a: np.ndarray                     # accel bias (3,)
    td: float = 0.0                     # time offset (s)
    role: str = "base"                  # "base" | "aux"
    topic: str | None = None            # source ROS topic (optional)
    gyro_only: bool = False             # aux calibrated gyro-only (lever/Da/Tg pinned)
    t_ga: float = 0.0                   # internal gyro<->accel time offset (s):
                                        # measured accel at stamp t = physical at t - t_ga


@dataclass
class MVIS:
    """Calibration for a single camera."""
    name: str                           # "cam0", "cam1", ...
    td: float                           # time offset (s)
    tr: float                           # rolling-shutter readout (s); 0 for global shutter
    model: str                          # distortion model, e.g. "radtan"
    K: np.ndarray                       # [fx, fy, cx, cy]
    dist: np.ndarray                    # distortion coeffs
    topic: str | None = None            # source ROS topic (optional)


@dataclass
class RigCalib:
    """Full rig calibration: base IMU + aux IMUs + cameras."""
    base_imu: ImuCalib
    aux_imus: list[ImuCalib] = field(default_factory=list)
    cams: list[MVIS] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Block parsers
# ---------------------------------------------------------------------------

def _imu_from_block(b: dict) -> ImuCalib:
    """Build an :class:`ImuCalib` from one ``imus[]`` YAML block.

    Base IMUs (``role == "base"`` or role absent) require ``imu_intr_xi`` — a
    complete result YAML always carries it for the calibrated base IMU. A
    missing field means the YAML was written by an old writer; we fail fast
    rather than silently substitute an identity model.

    Aux IMUs (``role == "aux"``) in this schema carry only extrinsics + td (the
    output writer emits no ``imu_intr_xi`` for them). When the field is absent
    we reconstruct an IDENTITY intrinsics model with zero bias — the honest
    behavior, since aux IMUs are re-stamped, not intrinsically corrected. If an
    aux block does carry ``imu_intr_xi`` we honor it.
    """
    role = b.get("role", "base")
    xi = b.get("imu_intr_xi")
    model = b.get("model")
    if xi is not None and model is None:
        # FAIL LOUD: imu_intr_xi's 24-dim tangent layout is model-dependent (calyx Dw is a
        # full 3x3 packed as 9; kalibr/rpng are triangular-6 + a free rotation). Reading a
        # calyx xi as the old default 'rpng' silently SCRAMBLES Dw/Da -> a wrong correction
        # that can still pass the accel-on-gravity check. Never guess the model.
        raise ValueError(
            f"IMU {b.get('name')!r} carries imu_intr_xi but no 'model' key. The xi tangent "
            "layout is model-dependent (calyx vs kalibr/rpng), so it cannot be read without "
            "the model — re-export the calib with 'model:' set (the writer emits it)."
        )
    if model is None:
        model = "rpng"          # only reached when xi is None -> identity model (layout moot)
    if xi is None:
        if role == "aux":
            intr = ImuModel.from_tangent(np.zeros(24), model)
        else:
            raise ValueError(
                f"IMU {b.get('name')!r} lacks imu_intr_xi; "
                "re-export the calib with the updated writer."
            )
    else:
        intr = ImuModel.from_tangent(np.array(xi, float), model)
    return ImuCalib(
        name=b.get("name", "imu0"),
        model=model,
        intr=intr,
        b_g=np.array(b.get("b_g", [0.0, 0.0, 0.0]), float),
        b_a=np.array(b.get("b_a", [0.0, 0.0, 0.0]), float),
        td=float(b.get("td", 0.0)),
        role=role,
        topic=b.get("topic"),
        gyro_only=bool(b.get("gyro_only", False)),
        t_ga=float(b.get("t_ga", 0.0)),
    )


def _cam_from_block(b: dict, index: int) -> MVIS:
    """Build a :class:`MVIS` from one ``cameras[]`` YAML block."""
    intr = b["intrinsics"]
    dist = b["distortion"]
    if is_removed_model(dist["model"]):
        raise ValueError(removed_model_message(dist["model"]))
    K = np.array([intr["fx"], intr["fy"], intr["cx"], intr["cy"]], float)
    return MVIS(
        name=b.get("name", f"cam{index}"),
        td=float(b.get("td", 0.0)),
        tr=float(b.get("rolling_shutter_readout", 0.0)),
        model=dist["model"],
        K=K,
        dist=np.array(dist.get("coeffs", []), float),
        topic=b.get("rostopic"),
    )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_rig_calib(path: str) -> RigCalib:
    """Load a MVIS result YAML and return a structured :class:`RigCalib`.

    The YAML is expected to have been written by
    :func:`mvis.io.cam_imu_output.write_cam_imu_calibration` (or
    equivalently ``yaml.safe_dump(build_cam_imu_dict(result), ...)``).
    """
    with open(path) as f:
        d = yaml.safe_load(f)

    imus = [_imu_from_block(b) for b in d.get("imus", [])]
    if not imus:
        raise ValueError("calibration YAML has no IMUs")
    base_imu = next((m for m in imus if m.role == "base"), imus[0])
    aux_imus = [m for m in imus if m is not base_imu]

    cams = [_cam_from_block(b, i) for i, b in enumerate(d.get("cameras", []))]

    return RigCalib(base_imu=base_imu, aux_imus=aux_imus, cams=cams)
