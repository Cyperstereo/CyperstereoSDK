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
"""Complete cam-IMU calibration output.

Serialize a `CamImuCalibResult` to ONE self-describing file carrying ALL params —
per-camera intrinsics+distortion+model, `T_cam_imu`, td, tr, reproj, stereo
baselines; base-IMU `Dw/Da/Tg` + biases; aux-IMU extrinsics+td — and export a
Kalibr `camchain-imucam` for interop. Round-trips as a camera-input (`io.camera_input`).
"""
from __future__ import annotations
import logging
import math

import numpy as np
import yaml

from mvis.core.camera_models import KALIBR_EXPORTABLE, dist_sanity_issue
from mvis.io.result_schema import calyx_dw_decomposed, camera_intrinsics_block, imu_diag

logger = logging.getLogger(__name__)


def _T_cam_imu(calib):
    """4x4 T_cam_imu (imu->cam) from CameraCalib.T (= T_CtoI, cam->imu)."""
    return np.linalg.inv(calib.T.matrix()).tolist()


def _img_size(e):
    sz = e.get("image_size")
    return [int(sz[0]), int(sz[1])] if sz is not None else None


def kalibr_export_issue(model, intr) -> str | None:
    """Why `intr` of `model` cannot be written as a Kalibr camchain, or None.
    By model (one outside `KALIBR_EXPORTABLE`) and by value (ucm alpha outside [0, 1):
    Kalibr's xi = alpha / (1 - alpha) is infinite or negative)."""
    if model not in KALIBR_EXPORTABLE:
        return f"camera model {model!r} has no Kalibr equivalent"
    if model == "ucm":
        issue = dist_sanity_issue("ucm", [float(intr[4])])
        if issue:
            return f"no Kalibr omni-none equivalent: {issue[1]}"
    return None


def kb4_kalibr_warning(intr, image_size=None, accounting=None) -> str | None:
    """What a kb4 camera loses as Kalibr pinhole-equi, or None when nothing.

    The same k1..k4 give the same pixel for every point with z > 0. Kalibr divides by z first, so a
    ray at or past 90 deg has no pixel there: a fit that used such corners keeps them only in MVIS.
    Kalibr has no fold check either: past a fold inside the image it maps pixels to the wrong ray,
    as MVIS's equidistant does."""
    from mvis.camera.kb_domain import kb_valid_domain
    notes = []
    th = (accounting or {}).get("max_theta_used_deg")
    if th is not None and th >= 90.0:
        notes.append(f"the fit used corners up to {th:.1f} deg off-axis, and Kalibr cannot "
                     f"project rays at or past 90 deg")
    info = kb_valid_domain(intr, image_size)
    if info["fold_theta_rad"] is not None and info["covers_image"] is False:
        notes.append(f"the polynomial folds at {math.degrees(info['fold_theta_rad']):.1f} deg, "
                     f"{info['max_radius_px']:.0f} px from the centre, inside the image, and "
                     f"Kalibr does not check for it")
    return "; ".join(notes) or None


def _kalibr_pack(model, intr):
    """Our flat cal_vec [fx,fy,cx,cy, dist...] -> Kalibr (camera_model,
    distortion_model, intrinsics, distortion_coeffs). Inverse of the camera-input
    `_from_kalibr` unpacking."""
    fx, fy, cx, cy = (float(intr[0]), float(intr[1]),
                      float(intr[2]), float(intr[3]))
    extra = [float(x) for x in intr[4:]]
    if model == "double_sphere":                   # Kalibr: [xi, alpha, fu, fv, pu, pv]
        return "ds", "none", extra[:2] + [fx, fy, cx, cy], []
    if model == "eucm":                            # Kalibr: [alpha, beta, fu, fv, pu, pv]
        return "eucm", "none", extra[:2] + [fx, fy, cx, cy], []
    if model not in KALIBR_EXPORTABLE:
        raise ValueError(f"camera model {model!r} has no Kalibr equivalent; "
                         f"no Kalibr camchain can be written for it")
    if model == "kb4":                             # same polynomial as pinhole-equi, z > 0 only
        return "pinhole", "equidistant", [fx, fy, cx, cy], extra
    if model == "ucm":                             # Kalibr omni-none: [xi, gu, gv, pu, pv]
        issue = kalibr_export_issue(model, intr)
        if issue:
            raise ValueError(issue)
        a = extra[0]                               # xi = a/(1-a), gamma = f/(1-a)
        return "omni", "none", [a / (1 - a), fx / (1 - a), fy / (1 - a), cx, cy], []
    if model == "omni_radtan":                     # intr = [fx,fy,cx,cy,xi,k1,k2,p1,p2]
        xi, k1, k2, p1 = extra[0], extra[1], extra[2], extra[3]
        p2 = extra[4] if len(extra) > 4 else 0.0   # legacy 8-long omni cal_vec
        return "omni", "radtan", [xi, fx, fy, cx, cy], [k1, k2, p1, p2]
    # radtan/equidistant: 4 coeffs; fov: the single [omega], which is exactly
    # Kalibr's pinhole-fov `distortion_coeffs` form (not zero-padded to 4)
    return "pinhole", model, [fx, fy, cx, cy], extra


def _cam_block(c, cal, e, topic=None):
    intr = np.asarray(cal.intr, float)
    blk = camera_intrinsics_block(c, cal.model, intr[0:4], intr[4:], _img_size(e), cal.tr)
    blk.update(                         # cam-IMU adds pose / td / reproj onto the shared core
        td=float(cal.td),
        T_cam_imu=_T_cam_imu(cal),
        reproj_rms_px=round(float(e.get("reproj_rms_px", float("nan"))), 4),
        **{k: round(float(e[k]), 4) for k in ("reproj_mean_px", "reproj_median_px", "reproj_std_px", "reproj_p90_px",
                                                  "reproj_p95_px", "reproj_max_px")
           if e.get(k) is not None},
        p_CtoC0_m=([float(v) for v in np.asarray(e["T_CtoC0"].translation(), float)]
                   if "T_CtoC0" in e else None),
    )
    if e.get("corner_accounting") is not None:
        blk["corner_accounting"] = e["corner_accounting"]
    if topic is not None:
        blk["rostopic"] = topic
    return blk


def _aux_imu_block(aux, model, topic=None) -> dict:
    """Serialize one aux-IMU `result.imus` entry to an `imus[]` YAML block.

    Emits `gyro_only: true` ONLY when the aux was calibrated gyro-only (so a
    reader knows lever/Da/Tg/accel-bias were pinned, not estimated); the flag
    is omitted for full-mode aux IMUs to avoid cluttering their blocks.
    """
    from mvis.imu import ImuModel
    a = dict(name=aux.get("name", "aux"), role="aux", model=model,
             td=float(aux.get("td", 0.0)))
    if "b_g_s" in aux:
        a["b_g"] = [float(x) for x in np.asarray(aux["b_g_s"], float)]
        a["b_a"] = [float(x) for x in np.asarray(aux["b_a_s"], float)]
    if aux.get("intr_xi24") is not None:
        axi = np.asarray(aux["intr_xi24"], float)
        ii = ImuModel.from_tangent(axi, model)
        a["imu_intr_xi"] = [float(v) for v in axi]
        a["Dw"] = ii.Dw().tolist(); a["Da"] = ii.Da().tolist()
        a["Dw_diag"], a["Da_diag"] = imu_diag(ii)
        if getattr(ii, "NAME", "").lower() == "calyx":
            a["Dw_decomposed"] = calyx_dw_decomposed(ii.Dw())
    if "T_IstoI" in aux:
        a["p_IsinI_m"] = [float(v) for v in np.asarray(aux["T_IstoI"].translation(), float)]
    if aux.get("t_ga") is not None:       # its own gyro<->accel offset; apply-calib shifts its accel
        a["t_ga"] = float(aux["t_ga"])
        if aux.get("t_ga_sigma") is not None:
            a["t_ga_sigma"] = float(aux["t_ga_sigma"])
    if aux.get("gyro_only"):
        a["gyro_only"] = True
    if topic is not None:
        a["topic"] = topic
    return a


def build_cam_imu_dict(result, cam_topics=None, imu_topics=None) -> dict:
    """MVIS-format dict with every cam-IMU parameter."""
    from mvis.imu import ImuModel, CALYX
    cams = result.cameras or [dict(calib=result.calib, reproj_rms_px=float("nan"))]
    cam_topics = cam_topics or []
    imu_topics = imu_topics or []
    out = {"cameras": [
        _cam_block(c, e["calib"], e,
                   topic=(cam_topics[c] if c < len(cam_topics) else None))
        for c, e in enumerate(cams)
    ]}

    model = getattr(result, "imu_model", "rpng")
    base = {"name": "imu0", "role": "base", "model": model,
            "b_g": [float(x) for x in np.asarray(result.b_g, float)],
            "b_a": [float(x) for x in np.asarray(result.b_a, float)]}
    if imu_topics:
        base["topic"] = imu_topics[0]
    if result.diagnostics.get("gravity") is not None:   # |g| the solve held fixed
        base["gravity"] = float(result.diagnostics["gravity"])
    if result.diagnostics.get("t_ga") is not None:   # gyro<->accel time offset
        base["t_ga"] = float(result.diagnostics["t_ga"])
        if result.diagnostics.get("t_ga_sigma") is not None:
            base["t_ga_sigma"] = float(result.diagnostics["t_ga_sigma"])
    xi = getattr(result, "imu_intr_xi", None)
    if xi is not None:
        xi = np.asarray(xi, float)
        ii = ImuModel.from_tangent(xi, model)   # absolute (identity ∘ ξ)
        base["imu_intr_xi"] = [float(v) for v in xi]   # full model (incl. Tg, R) — re-loadable
        base["Dw"] = ii.Dw().tolist()
        if getattr(ii, "NAME", "").lower() == "calyx":
            base["Dw_decomposed"] = calyx_dw_decomposed(ii.Dw())
        base["Da"] = ii.Da().tolist()
        base["Dw_diag"], base["Da_diag"] = imu_diag(ii)
        if ii.model == CALYX:   # report the post-hoc R_wtoI · U_w split
            R_wtoI, U_w = ii.decompose()
            base["R_wtoI"] = R_wtoI.tolist()
            base["U_w"] = U_w.tolist()
    imus = [base]
    for i, aux in enumerate(result.imus or []):
        aux_topic_idx = 1 + i
        topic = (imu_topics[aux_topic_idx]
                 if imu_topics and aux_topic_idx < len(imu_topics) else None)
        imus.append(_aux_imu_block(aux, model, topic=topic))
    out["imus"] = imus
    out["diagnostics"] = {"reproj_rms_px": round(float(result.diagnostics.get("reproj_rms_px", float("nan"))), 4),
                          "ok": bool(result.ok)}
    return out


def to_kalibr_camchain_imucam(result, cam_topics=None) -> dict:
    """Kalibr camchain-imucam export (cam0..; T_cam_imu, T_cn_cnm1, intrinsics).

    Keys are cam0.. in solve order, as Kalibr numbers them. `rostopic` is the topic the camera was
    read from (`cam_topics`), so a one-camera run on /cam1/... names /cam1/...; /cam<c>/image_raw
    only when no topics are given."""
    cams = result.cameras or []
    out = {}
    for c, e in enumerate(cams):
        cal = e["calib"]; intr = np.asarray(cal.intr, float)
        k = f"cam{c}"
        cam_m, dist_m, k_intr, k_coeffs = _kalibr_pack(cal.model, intr)
        if cal.model == "kb4" and (w := kb4_kalibr_warning(intr, _img_size(e),
                                                            e.get("corner_accounting"))):
            logger.warning(f"{k}: kb4 written to Kalibr as pinhole-equi with the same "
                           f"coefficients, exact in front of the camera, but {w}")
        out[k] = dict(
            camera_model=cam_m,
            distortion_model=dist_m,
            distortion_coeffs=k_coeffs,
            intrinsics=k_intr,
            resolution=_img_size(e),
            rostopic=(cam_topics[c] if cam_topics and c < len(cam_topics) and cam_topics[c]
                      else f"/cam{c}/image_raw"),
            T_cam_imu=_T_cam_imu(cal),
        )
        sz = _img_size(e)                       # Kalibr stores per-line readout
        if sz and sz[1] and float(cal.tr):      # line_delay = tr / rows (RS only)
            out[k]["line_delay"] = float(cal.tr) / int(sz[1])
        # Kalibr T_cn_cnm1 = point in cam_{c-1} -> cam_c. T_CtoC0 maps cam_c -> cam0,
        # so T_cn_cnm1 = (T_CtoC0[c])^-1 . T_CtoC0[c-1]  (cam0 frame cancels). cam0's
        # own T_CtoC0 is identity, so the c==1 case reduces to (T_CtoC0[1])^-1.
        if c > 0 and "T_CtoC0" in e:
            T_prev0 = cams[c - 1].get("T_CtoC0")
            T_c_to_cm1 = (e["T_CtoC0"].inverse().compose(T_prev0) if T_prev0 is not None
                         else e["T_CtoC0"].inverse())
            out[k]["T_cn_cnm1"] = T_c_to_cm1.matrix().tolist()
    return out


def write_cam_imu_calibration(result, path: str, kalibr_path: str | None = None,
                              cam_topics=None, imu_topics=None) -> None:
    """Write the complete MVIS cam-IMU yaml (+ optional Kalibr camchain-imucam)."""
    with open(path, "w") as f:
        yaml.safe_dump(build_cam_imu_dict(result, cam_topics=cam_topics,
                                          imu_topics=imu_topics), f, sort_keys=False)
    if kalibr_path:
        # build first: a non-exportable camera raises before the file is
        # opened, so no half-written camchain is left behind
        chain = to_kalibr_camchain_imucam(result, cam_topics=cam_topics)
        with open(kalibr_path, "w") as f:
            yaml.safe_dump(chain, f, sort_keys=False)
