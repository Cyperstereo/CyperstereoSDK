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
"""Shared serialization primitives for calibration results — one definition of the blocks that
both the cam-only (`output_writer`) and cam-IMU (`cam_imu_output`) writers emit, so the on-disk
schema is consistent across pipelines.
"""
from __future__ import annotations

from mvis.core.camera_models import KB_THETA_MODELS


def camera_intrinsics_block(cam_id, dist_model: str, K4, coeffs, image_size, tr) -> dict:
    """The per-camera intrinsics/distortion block shared by the cam-only and cam-IMU outputs:
    ``{id, model: pinhole_<dist>, intrinsics:{fx,fy,cx,cy}, distortion:{model, coeffs},
    image_size, rolling_shutter_readout}`` (+ ``valid_domain`` metadata for kb4, see
    `mvis.camera.kb_domain.kb_valid_domain`). Callers add pose/td/reproj fields on top. All numbers
    are coerced to plain Python floats so YAML/JSON stays clean."""
    fx, fy, cx, cy = (float(v) for v in list(K4)[:4])
    blk = {
        "id": cam_id,
        "model": f"pinhole_{dist_model}",
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "distortion": {"model": dist_model, "coeffs": [float(c) for c in coeffs]},
        "image_size": list(image_size),
        "rolling_shutter_readout": float(tr),
    }
    if dist_model in KB_THETA_MODELS:
        # metadata only (readers ignore it): where theta_d folds, vs the image
        from mvis.camera.kb_domain import kb_valid_domain     # lazy: heavy import
        blk["valid_domain"] = kb_valid_domain(
            [fx, fy, cx, cy] + [float(c) for c in coeffs], image_size)
    return blk


def imu_diag(ii, ndigits: int = 6):
    """``(Dw_diag, Da_diag)`` — the per-axis gyro/accel scale diagonals of an :class:`ImuModel`,
    each as three plain floats rounded to ``ndigits``. One definition for the extraction that the
    summary json (ndigits=5), the calib YAML, and the text report (ndigits=6) all hand-rolled."""
    dw = [round(float(ii.Dw()[i, i]), ndigits) for i in range(3)]
    da = [round(float(ii.Da()[i, i]), ndigits) for i in range(3)]
    return dw, da


def calyx_dw_qr(Dw):
    """``Dw = R · U`` for a CALYX gyro correction: (R, U) with R ∈ SO(3) the misalignment
    rotation and U upper-triangular with a positive diagonal (scale on the diagonal,
    cross-coupling above it). Unrounded; calyx_dw_decomposed and the report read from it."""
    import numpy as np
    A = np.asarray(Dw, float).reshape(3, 3)
    Q, U = np.linalg.qr(A)
    s = np.sign(np.diag(U))
    s[s == 0] = 1.0
    # diag(U) > 0, Q stays orthogonal. Element-wise on purpose: imu_model.decompose_dw
    # folds with diag(s) products, which drop the sign of exact zeros (-0.0 -> 0.0) and
    # so change the printed Dw_shear / rotvec for a Dw with exact zeros.
    return Q * s[None, :], U * s[:, None]


def calyx_dw_decomposed(Dw, ndigits: int = 5):
    """Physical reading of a CALYX full-3x3 gyro correction: ``Dw = R · U``
    (QR; R ∈ SO(3), U upper-triangular with positive diagonal). The calyx
    model has no separate gyro-frame rotation — misalignment lives inside
    Dw — so ``Dw_diag`` mixes rotation into the diagonal and is NOT a scale.
    Returns ``{Dw_scale, Dw_shear, gyro_misalign_rotvec}``: diag(U) = the
    per-axis gyro scale, the 3 upper off-diagonals of U (cross-coupling),
    and log(R) as a rotation vector in rad (project convention: rotations
    as log(R), never RPY)."""
    import numpy as np
    Q, U = calyx_dw_qr(Dw)
    # numpy-only SO(3) log (Rodrigues inverse; Q is near identity here)
    tr = float(np.trace(Q))
    c = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    theta = float(np.arccos(c))
    if theta < 1e-12:
        rv = np.zeros(3)
    else:
        w = np.array([Q[2, 1] - Q[1, 2], Q[0, 2] - Q[2, 0], Q[1, 0] - Q[0, 1]])
        rv = theta / (2.0 * np.sin(theta)) * w
    return {
        "Dw_scale": [round(float(U[i, i]), ndigits) for i in range(3)],
        "Dw_shear": [round(float(U[0, 1]), ndigits), round(float(U[0, 2]), ndigits),
                     round(float(U[1, 2]), ndigits)],
        "gyro_misalign_rotvec": [round(float(v), 6) for v in rv],
    }


def cam_imu_sensor_rows(result, *, require_sigma: bool) -> dict:
    """The unrounded per-sensor quantities of a cam-IMU result that both the CLI summary
    (``cam_imu.json``) and the cam-IMU report rows start from; each caller keeps its own
    rounding, units and extra fields. Returns ``{cams, base, aux}``:

    * ``cams``: per ``result.cameras`` entry ``{entry, calib, p_m, rotvec, baseline_m}`` with
      the camera position in the IMU frame (m), ``log(R)`` of ``T_CtoI`` (rad) and, for a
      camera after the first that has ``T_CtoC0``, its translation (m), else None.
    * ``base``: ``{model, intr}``, the base-IMU ``ImuModel`` built from ``imu_intr_xi``, or
      None when there is none (or, with ``require_sigma``, when ``imu_intr_sigma`` is None,
      i.e. the intrinsics were pinned).
    * ``aux``: per ``result.imus`` entry ``{entry, p_m, rotvec, intr}``: the ``T_IstoI``
      translation (m) and ``log(R)`` (rad) when present, and the aux ``ImuModel`` when its
      24-vector was freed (nonzero), else None.
    """
    import numpy as np
    from mvis.calibration.factors.lie_groups import log_so3
    from mvis.imu import ImuModel
    cams = []
    for c, e in enumerate(result.cameras or []):
        cal = e["calib"]
        cams.append(dict(
            entry=e, calib=cal,
            p_m=np.asarray(cal.T.translation(), float),
            rotvec=log_so3(cal.T.rotation().matrix()),
            baseline_m=(np.asarray(e["T_CtoC0"].translation(), float)
                        if c > 0 and "T_CtoC0" in e else None)))
    model = getattr(result, "imu_model", "rpng")
    xi = getattr(result, "imu_intr_xi", None)
    base_ii = None
    if xi is not None and not (require_sigma and result.imu_intr_sigma is None):
        base_ii = ImuModel.from_tangent(np.asarray(xi, float), model)
    aux = []
    for m in getattr(result, "imus", None) or []:
        row = dict(entry=m, p_m=None, rotvec=None, intr=None)
        if "T_IstoI" in m:
            T = m["T_IstoI"]
            row["p_m"] = np.asarray(T.translation(), float)
            row["rotvec"] = log_so3(T.rotation().matrix())
        xi24 = m.get("intr_xi24")
        if xi24 is not None:
            xi24 = np.asarray(xi24, float)
            if xi24.size >= 12 and np.max(np.abs(xi24)) > 1e-9:     # freed
                row["intr"] = ImuModel.from_tangent(xi24, model)
        aux.append(row)
    return dict(cams=cams, base=dict(model=model, intr=base_ii), aux=aux)
