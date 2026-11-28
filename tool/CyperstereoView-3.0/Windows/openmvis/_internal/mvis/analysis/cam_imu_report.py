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
"""Cam-IMU analysis report.

Mirrors the camera-calib `analysis/` reports for the joint cam-IMU result:
per-camera reproj + lever/rotation/td/tr + stereo baseline (with σ), base-IMU
Dw/Da diagonals + biases (+ drift), aux-IMU extrinsics/td, solve diagnostics, and
an optional **vs-reference** section (lever Δ, rotation Δ, baseline Δ — what we did
by hand for MIC / TUM-VI). Text + self-contained HTML.

`reference` (optional) — published/known values to diff against::

    {"label": "Monado Valve Index",
     "baseline_mm": 134.0,                 # |cam1 -> cam0|, magnitude OR [x,y,z]
     "lever_mm": {0: [..], 1: [..]},       # per-cam camera origin in IMU frame
     "rotation_rad": {0: 1.5708}}          # per-cam |T_CtoI| rotation angle, rad
                                           # (or "rotation_deg" in degrees, still read)

Rotations are reported as the log(R) rotation vector and its norm in rad: the
rows carry ``rot_rad``, ``rot_angle_rad``, ``rot_sig_rad`` (aux IMUs ``rot_rad``,
``rot_angle_rad``; a calyx base IMU ``R_wtoI_rad``) and the reference deltas
``rotation_delta_rad``.
"""
from __future__ import annotations

import html
import logging

import numpy as np

from mvis.calibration.factors.lie_groups import log_so3
from mvis.core.camera_models import KB_THETA_MODELS, dist_names
from mvis.core.gravity import describe as describe_gravity
from mvis.imu import CALYX
from mvis.calibration.reproj_stats import corner_stats
from mvis.io.result_schema import calyx_dw_qr, cam_imu_sensor_rows, imu_diag


def _cam_rows(result):
    """Per-camera derived quantities (lever, rotation, td/tr, reproj, baseline)."""
    rows = []
    for c, cr in enumerate(cam_imu_sensor_rows(result, require_sigma=True)["cams"]):
        e, cal = cr["entry"], cr["calib"]
        sig = np.asarray(e.get("calib_sigma", np.zeros(cal.DIM)), float)
        rot = cr["rotvec"]                                  # rad, body axes
        p = cr["p_m"]                                       # camera in IMU (m)
        row = dict(
            id=c, model=cal.model, is_fisheye=bool(cal.is_fisheye),
            fx=float(cal.intr[0]), fy=float(cal.intr[1]),
            cx=float(cal.intr[2]), cy=float(cal.intr[3]),
            dist=[float(x) for x in cal.intr[4:]],
            # coefficient names for omni (xi k1 k2 p1 p2) / kb4 (k1..kN) /
            # ucm (alpha); other models unlabeled
            dist_names=dist_names(cal.model),
            lever_mm=(1e3 * p).tolist(),
            lever_norm_mm=float(1e3 * np.linalg.norm(p)),
            lever_sig_mm=(1e3 * sig[3:6]).tolist(),
            rot_rad=np.asarray(rot, float).tolist(),
            rot_angle_rad=float(np.linalg.norm(rot)),
            rot_sig_rad=sig[0:3].tolist(),
            td_ms=float(1e3 * cal.td), td_sig_ms=float(1e3 * sig[6]),
            tr_ms=float(1e3 * cal.tr), tr_pinned=bool(sig[7] < 1e-6),
            intr_sig=(sig[8:12].tolist() if sig.size >= 12 else None),
            dist_sig=(sig[12:].tolist() if sig.size > 12 else None),
            reproj_rms_px=float(e.get("reproj_rms_px", float("nan"))),
            reproj_median_px=e.get("reproj_median_px"),
            reproj_p95_px=e.get("reproj_p95_px"),
            reproj_mean_px=e.get("reproj_mean_px"),
            reproj_std_px=e.get("reproj_std_px"),
            reproj_p90_px=e.get("reproj_p90_px"),
            reproj_max_px=e.get("reproj_max_px"),
            n_frames=int(e.get("n_frames", 0)),
            valid_domain=_kb_domain(cal, e),
            corner_accounting=e.get("corner_accounting"),
        )
        if cr["baseline_m"] is not None:
            b = cr["baseline_m"]
            row["baseline_mm"] = (1e3 * b).tolist()
            row["baseline_norm_mm"] = float(1e3 * np.linalg.norm(b))
            if e.get("T_CtoC0_sigma") is not None:          # [rot(3), trans(3)] of the joint marginal
                row["baseline_sig_mm"] = (1e3 * np.asarray(e["T_CtoC0_sigma"], float)[3:6]).tolist()
        try:
            from mvis.analysis.views.geometry import fov_deg
            from mvis.calibration.factors.camera_projection import k_cal_vec
            size = e.get("image_size")
            if size:
                intr = np.asarray(cal.intr, float)
                row["fov"] = fov_deg(cal.model, k_cal_vec(cal.model, intr[:4], intr[4:]), size)
                row["image_size"] = [int(size[0]), int(size[1])]
        except Exception:                                   # a model the views can't trace: no FOV row
            pass
        rows.append(row)
    return rows


def _kb_domain(cal, e):
    """kb4: the fit's valid domain (pipeline-computed, else recomputed), or None."""
    if cal.model not in KB_THETA_MODELS:
        return None
    if e.get("valid_domain") is not None:
        return e["valid_domain"]
    from mvis.camera.kb_domain import kb_valid_domain
    return kb_valid_domain(cal.intr, e.get("image_size"))


def _kb_domain_text(row) -> str | None:
    """One-line valid-domain summary for a kb4 camera row, or None."""
    info = row.get("valid_domain")
    if info is None:
        return None
    from mvis.camera.kb_domain import format_kb_valid_domain, kb_domain_tag
    tag = kb_domain_tag(info)
    return format_kb_valid_domain(info, row["model"]) + (f"  [{tag}]" if tag else "")


def _corner_accounting_html(row) -> str:
    from mvis.calibration.corner_accounting import accounting_text
    return accounting_text(row["corner_accounting"])


def _dist_label(row) -> str:
    """' (xi k1 k2 p1 p2)' for a row carrying coefficient names, else ''."""
    names = row.get("dist_names")
    return f" ({' '.join(names)})" if names else ""


def _dist_values(row) -> str:
    """Space-joined distortion coefficients."""
    return " ".join("{:.5f}".format(x) for x in row["dist"])


def _imu_block(result):
    """Base-IMU biases + Dw/Da diagonals; aux-IMU extrinsics/td."""
    sensors = cam_imu_sensor_rows(result, require_sigma=True)
    base = dict(name="imu0", role="base",
                b_g=np.asarray(result.b_g, float).tolist(),
                b_a=np.asarray(result.b_a, float).tolist(),
                b_g_drift=np.asarray(result.b_g_drift, float).tolist(),
                b_a_drift=np.asarray(result.b_a_drift, float).tolist(),
                pinned=result.imu_intr_sigma is None,
                t_ga=result.diagnostics.get("t_ga"),
                t_ga_sigma=result.diagnostics.get("t_ga_sigma"))
    model = sensors["base"]["model"]
    base["imu_model"] = model
    ii = sensors["base"]["intr"]
    if ii is not None:
        base["Dw_diag"], base["Da_diag"] = imu_diag(ii)
        base["Dw"], base["Da"] = ii.Dw().tolist(), ii.Da().tolist()
        base["Tg"] = ii.Tg().tolist()
        if ii.model == CALYX:
            R_wtoI, _ = ii.decompose()
            base["R_wtoI_rad"] = np.asarray(log_so3(R_wtoI), float).tolist()
            base["Dw_decomp"] = _dw_decomp(ii.Dw())
    aux = []
    for ar in sensors["aux"]:
        m = ar["entry"]
        a = dict(name=m.get("name", "aux"), td_ms=float(1e3 * m.get("td", 0.0)))
        if m.get("t_ga") is not None:              # aux IMU's own gyro<->accel offset
            a["t_ga"] = float(m["t_ga"])
            a["t_ga_sigma"] = m.get("t_ga_sigma")
        if "chi2_dof" in m:
            a["chi2_dof"] = float(m["chi2_dof"])
        if ar["p_m"] is not None:
            a["p_IsinI_mm"] = (1e3 * ar["p_m"]).tolist()
            rot = ar["rotvec"]
            a["rot_rad"] = np.asarray(rot, float).tolist()
            a["rot_angle_rad"] = float(np.linalg.norm(rot))
        if "b_g_s" in m:
            a["b_g"] = np.asarray(m["b_g_s"], float).tolist()
        if "b_a_s" in m:
            a["b_a"] = np.asarray(m["b_a_s"], float).tolist()
        if ar["intr"] is not None:                 # freed
            ii = ar["intr"]
            a["Dw_diag"], a["Da_diag"] = imu_diag(ii)
            a["Dw"], a["Da"] = ii.Dw().tolist(), ii.Da().tolist()
            if ii.model == CALYX:
                a["Dw_decomp"] = _dw_decomp(ii.Dw())
            a["Tg"] = ii.Tg().tolist()
        aux.append(a)
    return base, aux


def _rotvec_matrix(rot_rad):
    """R from a rotation vector in rad (the rows' `rot_rad`)."""
    from mvis.calibration.factors.lie_groups import exp_so3
    return np.asarray(exp_so3(np.asarray(rot_rad, float)), float)


def _reference_angle_rad(reference, cam_id):
    """|T_CtoI| reference angle in rad for one camera, or None.

    Per camera, a ``rotation_rad`` entry wins, else a ``rotation_deg`` entry is converted. Read
    lazily so that None, non-scalar or other cameras' entries are ignored, as before.
    """
    from mvis.analysis.reference import camera_value
    for key, to_rad in (("rotation_rad", float), ("rotation_deg", lambda v: float(np.radians(float(v))))):
        v = camera_value(reference.get(key), cam_id)     # camN / N / 'N' keys (F3)
        if v is not None and np.ndim(v) == 0:
            return to_rad(v)
    return None


def _ref_deltas(rows, reference):
    """vs-reference diffs (lever Δ / rotation Δ / baseline Δ) where provided."""
    if not reference:
        return None
    from mvis.analysis.reference import reference_deltas
    parsed = reference_deltas(reference)       # camN / N / 'N' keys, scalar or vector baseline
    out = {"label": parsed["label"], "cams": [], "baseline": None}
    lever_ref = parsed["lever_mm"]
    full = None
    if reference.get("camchain") or reference.get("rotvec"):   # full poses: lever and rotation differences
        try:
            from mvis.analysis.views.spec import reference_rig
            full = reference_rig(reference, [_rotvec_matrix(r["rot_rad"]) for r in rows], len(rows))
        except Exception:
            full = None
    for k, r in enumerate(rows):
        d = {"id": r["id"]}
        fr = full["cams"][k] if full and k < len(full["cams"]) else None
        if fr is not None:
            d["lever_delta_mm"] = float(np.linalg.norm(np.asarray(r["lever_mm"]) - np.asarray(fr["p"])))
            if fr["R"] is not None:
                Rm = _rotvec_matrix(r["rot_rad"]) @ np.asarray(fr["R"]).reshape(3, 3).T
                d["rotation_delta_rad"] = float(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1.0, 1.0)))
        elif r["id"] in lever_ref:
            lr = np.asarray(lever_ref[r["id"]], float)
            d["lever_delta_mm"] = float(np.linalg.norm(np.asarray(r["lever_mm"]) - lr))
        ref_angle = None if "rotation_delta_rad" in d else _reference_angle_rad(reference, r["id"])
        if ref_angle is not None:
            d["rotation_delta_rad"] = float(abs(r["rot_angle_rad"] - ref_angle))
        if len(d) > 1:
            out["cams"].append(d)
    ref_norm = parsed["baseline_mm"]
    bcam = next((r for r in rows if "baseline_norm_mm" in r), None)
    if ref_norm is not None and bcam is not None:
        out["baseline"] = {"ref_mm": ref_norm, "got_mm": bcam["baseline_norm_mm"],
                           "delta_mm": abs(bcam["baseline_norm_mm"] - ref_norm)}
    return out


# ----------------------------------------------------------- summary parameter folds
def _config_solve(config) -> dict:
    """The run's solve settings as a dict, from a config dict, object or its YAML source text."""
    if config is None:
        return {}
    if isinstance(config, str):
        import yaml
        try:
            config = yaml.safe_load(config) or {}
        except Exception:
            return {}
    solve = config.get("solve") if isinstance(config, dict) else getattr(config, "solve", None)
    if solve is None:
        return {}
    return solve if isinstance(solve, dict) else {k: getattr(solve, k) for k in dir(solve) if k.startswith("sigma")}


def _calib_yaml(result_files) -> dict | None:
    """The run's cam_imu_calib.yaml, for each sensor's Copy block; None when not among the result files."""
    import os
    import yaml
    for pth in result_files or []:
        if str(pth).endswith("_calib.yaml") and os.path.exists(pth):
            try:
                return yaml.safe_load(open(pth))
            except Exception:
                return None
    return None


def _fov_text(fov) -> str | None:
    if not fov or fov.get("h") is None or fov.get("v") is None:
        return None
    return f"{fov['h']:.0f}° × {fov['v']:.0f}°"


def _cam_fold(r, calib_doc) -> str:
    from mvis.analysis.param_folds import group, num, sensor_fold, val, vec, yaml_block
    names = r.get("dist_names") or [f"d{i}" for i in range(len(r["dist"]))]
    dsig = r.get("dist_sig") or [None] * len(r["dist"])
    isig = r.get("intr_sig") or [None] * 4
    med, p95 = r.get("reproj_median_px"), r.get("reproj_p95_px")
    key = (f"{r['model']} · fx {num(r['fx'], 'px2')} · td {num(r['td_ms'], 'ms')} ms · lever "
           f"{num(r['lever_norm_mm'], 'mm')} mm"
           + (f" · reproj median / P95 {num(med, 'px')} / {num(p95, 'px')} px" if med is not None else ""))
    groups = [
        group("Intrinsics", [("model", r["model"]),
                             ("fx", val(r["fx"], "px2", isig[0])), ("fy", val(r["fy"], "px2", isig[1])),
                             ("cx", val(r["cx"], "px2", isig[2])), ("cy", val(r["cy"], "px2", isig[3])),
                             *[(n, val(v, "coef", s)) for n, v, s in zip(names, r["dist"], dsig)],
                             ("image size", f"{r['image_size'][0]} × {r['image_size'][1]}" if r.get("image_size") else None),
                             ("field of view H × V", _fov_text(r.get("fov")))]),
        group("Time", [("td (t_imu = t_cam + td)", val(r["td_ms"], "ms", r["td_sig_ms"])),
                       ("readout", "global shutter" if r["tr_pinned"] and abs(r["tr_ms"]) < 1e-3   # < 1 µs
                        else val(r["tr_ms"], "ms") + (" (pinned)" if r["tr_pinned"] else ""))]),
        group("Camera in IMU (T_C→I)", [("lever x, y, z (mm)", vec(r["lever_mm"], "mm", r["lever_sig_mm"])),
                                        ("|lever|", val(r["lever_norm_mm"], "mm")),
                                        ("log(R) x, y, z (rad)", vec(r["rot_rad"], "rad", r["rot_sig_rad"])),
                                        ("|rotation|", val(r["rot_angle_rad"], "rad"))]),
        group(f"Rig (cam{r['id']} → cam0)", [("baseline x, y, z (mm)", vec(r["baseline_mm"], "mm", r.get("baseline_sig_mm"))),
                                            ("|baseline|", val(r["baseline_norm_mm"], "mm"))]
              if "baseline_mm" in r else []),
        group("Reprojection", [("mean", val(r.get("reproj_mean_px"), "px")), ("median", val(med, "px")),
                               ("std", val(r.get("reproj_std_px"), "px")), ("P90", val(r.get("reproj_p90_px"), "px")),
                               ("P95", val(p95, "px")), ("max", val(r.get("reproj_max_px"), "px")),
                               ("frames", f"{r['n_frames']:,}" if r.get("n_frames") else None)]),
    ]
    blk = next((c for c in (calib_doc or {}).get("cameras", []) if c.get("id") == r["id"]), None)
    return sensor_fold(f"cam{r['id']}", key, groups, yaml_block(blk))


def _imu_fold(m, role: str, solve: dict, chi2, calib_doc, idx: int) -> str:
    from mvis.analysis.param_folds import group, num, sensor_fold, val, vec, yaml_block
    dd = m.get("Dw_decomp")
    model = m.get("imu_model", "—" if role == "base" else "")
    key_parts = [model] if model else []
    if m.get("t_ga") is not None:
        key_parts.append(f"t_ga {num(1e3 * m['t_ga'], 'ms')} ms")
    if dd:
        key_parts.append("gyro scale " + " / ".join(f"{v:.3f}" for v in dd["scale"]))
    elif m.get("Dw_diag"):
        key_parts.append("Dw diag " + " / ".join(f"{v:.3f}" for v in m["Dw_diag"]))
    if chi2 is not None:
        key_parts.append(f"χ²/dof {chi2:.3f}")
    if role != "base" and m.get("td_ms") is not None:
        key_parts.append(f"td {num(m['td_ms'], 'ms')} ms")
    t_ga = m.get("t_ga")
    t_ga_sig = m.get("t_ga_sigma")
    groups = [
        group("Model", [("IMU model", model or None),
                        ("intrinsics", "pinned" if m.get("pinned") else ("estimated" if m.get("Dw") else None))]),
        group("Biases", [("b_g (first state, rad/s)", vec(m.get("b_g"), "rads") if m.get("b_g") else None),
                         ("b_g drift (rad/s)", vec(m.get("b_g_drift"), "rads") if m.get("b_g_drift") else None),
                         ("b_a (first state, m/s²)", vec(m.get("b_a"), "mps2") if m.get("b_a") else None),
                         ("b_a drift (m/s²)", vec(m.get("b_a_drift"), "mps2") if m.get("b_a_drift") else None)]),
        group("Gyro", [("Dw diag", vec(m["Dw_diag"], "scale") if m.get("Dw_diag") else None),
                       ("scale diag(U)", vec(dd["scale"], "scale") if dd else None),
                       ("shear U01 / U02 / U12", vec(dd["shear"], "coef") if dd else None),
                       ("misalignment log(R) (rad)", vec(dd["rotvec_rad"], "rad") if dd else None),
                       ("|misalignment|", val(dd["angle_rad"], "rad") if dd else None)]),
        group("Accel", [("Da diag", vec(m["Da_diag"], "scale") if m.get("Da_diag") else None),
                        ("R_wtoI log(R) (rad)", vec(m["R_wtoI_rad"], "rad") if m.get("R_wtoI_rad") else None)]),
        group("Time", [("t_ga gyro → accel", val(1e3 * t_ga, "ms", 1e3 * t_ga_sig if t_ga_sig else None)
                        if t_ga is not None else None),
                       ("td (aux, to imu0)", val(m.get("td_ms"), "ms") if role != "base" else None)]),
        group("In base IMU (T_Is→I)", [("lever x, y, z (mm)", vec(m["p_IsinI_mm"], "mm")),
                                       ("log(R) x, y, z (rad)", vec(m["rot_rad"], "rad")),
                                       ("|rotation|", val(m["rot_angle_rad"], "rad"))]
              if role != "base" and m.get("p_IsinI_mm") is not None else []),
        group("Noise used (solve config)", [(k, f"{float(solve[k]):.3g}") for k in ("sigma_g", "sigma_bg", "sigma_a", "sigma_ba")
                                            if role == "base" and solve.get(k) is not None]
              + ([("IMU χ²/dof", f"{chi2:.3f}")] if chi2 is not None else [])),
    ]
    imus = (calib_doc or {}).get("imus") or []
    blk = imus[idx] if idx < len(imus) else None
    return sensor_fold(f"{m.get('name', 'imu')} ({role})", " · ".join(key_parts), groups, yaml_block(blk))


def _param_folds_html(rows, base, aux, d, config, result_files) -> str:
    """Summary: every calibrated parameter, one closed fold per sensor (cameras, base IMU, aux IMUs)."""
    from mvis.analysis.param_folds import folds_block
    doc = _calib_yaml(result_files)
    solve = _config_solve(config)
    folds = [_cam_fold(r, doc) for r in rows]
    folds.append(_imu_fold(base, "base", solve, d.get("imu_chi2_dof"), doc, 0))
    folds += [_imu_fold(a, "aux", {}, a.get("chi2_dof"), doc, i + 1) for i, a in enumerate(aux)]
    return folds_block(folds)


# --------------------------------------------------------------------------- text
def _text_reproj_line(r) -> str:
    """One camera's reprojection line: mean / median / std (Kalibr's per-camera three) / p90 / p95 /
    max over all corners where the result has them, else median / p95, else RMS."""
    n = f"({r['n_frames']} frames)"
    if r.get("reproj_mean_px") is not None and r.get("reproj_std_px") is not None:
        tail = (f" / p90 {r['reproj_p90_px']:.3f}" if r.get("reproj_p90_px") is not None else "") + \
               f" / p95 {r['reproj_p95_px']:.3f}" + \
               (f" / max {r['reproj_max_px']:.2f}" if r.get("reproj_max_px") is not None else "")
        return (f"  reproj mean {r['reproj_mean_px']:.3f} / med {r['reproj_median_px']:.3f} / std "
                f"{r['reproj_std_px']:.3f}{tail} px  {n}")
    if r.get("reproj_median_px") is not None:
        return f"  reproj med {r['reproj_median_px']:.3f} / p95 {r['reproj_p95_px']:.3f} px  {n}"
    return f"  reproj {r['reproj_rms_px']:.3f} px  {n}"


def _text_dw_decomp(dd) -> list[str]:
    """A calyx Dw's R · U split (Dw_diag mixes the rotation in, so it is not the scale)."""
    return [f"  Dw = R·U (calyx): scale diag(U) " + " ".join(f"{x:.5f}" for x in dd["scale"])
            + "   shear U01/U02/U12 " + " ".join(f"{x:+.5f}" for x in dd["shear"]),
            f"  gyro misalignment log(R) (rad): " + " ".join(f"{x:+.6f}" for x in dd["rotvec_rad"])
            + f"  |{dd['angle_rad']:.6f}|"]


def cam_imu_text_report(result, reference=None) -> str:
    rows = _cam_rows(result)
    base, aux = _imu_block(result)
    d = result.diagnostics
    L = ["=" * 64, "OpenMVIS — Joint Camera-IMU Calibration Report", "=" * 64,
         f"status: {'OK' if result.ok else 'FAILED'}   "
         + (f"reproj med {d['reproj_median_px']:.3f} / p95 "
            f"{d['reproj_p95_px']:.3f} px   "
            if d.get('reproj_median_px') is not None else
            f"reproj RMS {d.get('reproj_rms_px', float('nan')):.3f} px   ")
         + f"states {d.get('n_states', 0)}  iters {d.get('iterations', 0)}",
         f"chi2/dof  cam {d.get('cam_chi2_dof', float('nan')):.2f}   "
         f"imu {d.get('imu_chi2_dof', float('nan')):.2f}"]
    if d.get("gravity") is not None:
        L.append("gravity: " + describe_gravity(float(d["gravity"]), d.get("gravity_source", "default"),
                                                d.get("gravity_site"), d.get("gravity_place")))
    for r in rows:
        L += ["", f"cam{r['id']}  ({r['model']})",
              _text_reproj_line(r),
              f"  intr: fx {r['fx']:.2f} fy {r['fy']:.2f} cx {r['cx']:.2f} cy {r['cy']:.2f}",
              f"  dist{_dist_label(r)}: " + _dist_values(r),
              f"  td: {r['td_ms']:+.3f} ms (σ {r['td_sig_ms']:.3f})   "
              f"tr: {r['tr_ms']:.3f} ms{' [pinned]' if r['tr_pinned'] else ''}",
              f"  T_CtoI lever (mm): [{', '.join(f'{v:.2f}' for v in r['lever_mm'])}]  "
              f"|{r['lever_norm_mm']:.2f}|  (σ mm [{', '.join(f'{v:.2f}' for v in r['lever_sig_mm'])}])",
              f"  T_CtoI rot (rad): [{', '.join(f'{v:.5f}' for v in r['rot_rad'])}]  "
              f"|{r['rot_angle_rad']:.5f}|"]
        if _kb_domain_text(r):
            L.append(f"  {_kb_domain_text(r)}")
        if r.get("corner_accounting"):
            from mvis.calibration.corner_accounting import accounting_text
            L.append(f"  {accounting_text(r['corner_accounting'])}")
        if "baseline_mm" in r:
            L.append(f"  baseline cam{r['id']}->cam0 (mm): "
                     f"[{', '.join(f'{v:.2f}' for v in r['baseline_mm'])}]  |{r['baseline_norm_mm']:.2f}|")
    L += ["", "IMU imu0 (base)",
          f"  b_g: " + " ".join(f"{x:.5f}" for x in base["b_g"]) +
          "   drift " + " ".join(f"{x:.5f}" for x in base["b_g_drift"]),
          f"  b_a: " + " ".join(f"{x:.4f}" for x in base["b_a"]) +
          "   drift " + " ".join(f"{x:.4f}" for x in base["b_a_drift"])]
    if base["pinned"]:
        L.append("  Dw/Da: pinned (phase A)")
    else:
        L += [f"  Dw_diag (gyro scale):  " + " ".join(f"{x:.5f}" for x in base["Dw_diag"]),
              f"  Da_diag (accel scale): " + " ".join(f"{x:.5f}" for x in base["Da_diag"])]
        if base.get("Dw_decomp"):
            L += _text_dw_decomp(base["Dw_decomp"])
    if base.get("t_ga") is not None:
        sig = base.get("t_ga_sigma")
        L.append(f"  t_ga gyro->accel: {base['t_ga'] * 1e3:+.3f} ms"
                 + (f" ± {sig * 1e3:.3f}" if sig else "")
                 + "   (gyro clock is reference: accel stamped t sampled at t - t_ga)")
    for a in aux:
        L.append("")
        head = f"IMU {a['name']} (aux): td {a['td_ms']:+.3f} ms"
        if a.get("t_ga") is not None:
            head += f"   t_ga {a['t_ga'] * 1e3:+.3f} ms"
        if "chi2_dof" in a:
            head += f"   chi2/dof {a['chi2_dof']:.2f}"
        L.append(head)
        if "p_IsinI_mm" in a:
            L.append("  T_IstoI p (mm): [" + ", ".join(f"{v:.2f}" for v in a["p_IsinI_mm"]) +
                     f"]  |{np.linalg.norm(a['p_IsinI_mm']):.2f}|")
        if "rot_rad" in a:
            L.append("  T_IstoI rot (rad): [" + ", ".join(f"{v:.5f}" for v in a["rot_rad"]) +
                     f"]  |{a['rot_angle_rad']:.5f}|")
        if "b_g" in a:
            L.append("  b_g: " + " ".join(f"{x:.5f}" for x in a["b_g"]))
        if "b_a" in a:
            L.append("  b_a: " + " ".join(f"{x:.4f}" for x in a["b_a"]))
        if "Dw_diag" in a:
            L.append("  Dw_diag: " + " ".join(f"{x:.5f}" for x in a["Dw_diag"]) +
                     "   Da_diag: " + " ".join(f"{x:.5f}" for x in a["Da_diag"]))
            if a.get("Dw_decomp"):
                L += _text_dw_decomp(a["Dw_decomp"])
    deltas = _ref_deltas(rows, reference)
    if deltas:
        L += ["", f"vs reference: {deltas['label']}"]
        for c in deltas["cams"]:
            parts = []
            if "lever_delta_mm" in c:
                parts.append(f"lever Δ {c['lever_delta_mm']:.2f} mm")
            if "rotation_delta_rad" in c:
                parts.append(f"rotation Δ {c['rotation_delta_rad']:.5f} rad")
            L.append(f"  cam{c['id']}: " + "  ".join(parts))
        if deltas["baseline"]:
            b = deltas["baseline"]
            L.append(f"  baseline: {b['got_mm']:.2f} mm vs {b['ref_mm']:.2f} mm "
                     f"(Δ {b['delta_mm']:.2f} mm)")
    L.append("=" * 64)
    return "\n".join(L)


# --------------------------------------------------------------------------- html
from mvis.analysis.report_shell import (  # shared shell  # noqa: E402
    badge as _badge, cam_label, graded, html_table as _tbl, render_page, report_plot_style, stat_box as _stat)
from mvis.analysis.verdict import GRADE_DISPLAY, grade  # noqa: E402


def _grade(value, good, marginal):
    """Map a metric to a display grade (lower is better): good <= good, marginal <= marginal."""
    return GRADE_DISPLAY.get(grade(value, good, marginal))


# Reprojection grading thresholds (px) in corner-norm (Kalibr) units. The
# original per-axis thresholds (median 1.0/2.0, p95 2.0/4.0) are scaled by
# sqrt(2) — same residuals read sqrt(2) higher as corner norms — so a grade
# means the same fit quality as before the 2026-08-29 convention switch.
# Single source: mvis.analysis.verdict (shared with `mvis verdict`).
from mvis.analysis.verdict import MED_GRADE as _MED_GRADE, P95_GRADE as _P95_GRADE, \
    RMS_GRADE as _RMS_GRADE, CHI2_GRADE as _CHI2_GRADE  # noqa: E402


# rotation-vs-reference grade: good <= 0.5 deg, marginal <= 2 deg, in rad
_ROT_DELTA_GRADE_RAD = (float(np.radians(0.5)), float(np.radians(2.0)))

_MAX_SCENE_STEPS = 600        # progress-timeline states (page size); see _cam_views


def _cam_views(result, board, cam_data):
    """Every camera riding the IMU, for the scene's selector / panel / co-vis.

    Returns (state_idx, views): the progress timeline uses the states
    `state_idx` (thinned to <= _MAX_SCENE_STEPS); camera c's view carries
    its T_CtoI, per-step frustum images, side panel and detected point ids.
    Camera c at state k: pose T_BinW^-1 . X(k) . T_CtoI_c, frame nearest
    t_k - td_c (capture time on the IMU clock is t_frame + td); coverage
    counts every frame since the previous step. Without per-camera data
    (`cam_data[c]` = dict(frames, image_size, thumb_dir)) a view has just its
    T_CtoI; ([], None) when there is no camera or trajectory.
    """
    from types import SimpleNamespace
    import gtsam
    from mvis.analysis.pose_images import match_frames, pose_images, pose_panel
    cams = getattr(result, "cameras", None) or []
    values = getattr(result, "values", None)
    if not cams or values is None:
        return [], None
    X = gtsam.symbol_shorthand.X
    n = 0
    while values.exists(X(n)):
        n += 1
    keep = np.unique(np.linspace(0, n - 1, min(n, _MAX_SCENE_STEPS)).round()
                     .astype(int)).tolist() if n else []
    T_BinW = getattr(result, "T_BinW", None)
    t_states = getattr(result, "t_states", None)
    have_t = t_states is not None and len(t_states) == n
    views = []
    for c, e in enumerate(cams):
        cal = e["calib"]
        T = cal.T.matrix()
        view = dict(name=f"cam{c}", cam=(T[:3, :3], T[:3, 3]))
        cd = (cam_data or [])[c] if c < len(cam_data or []) else None
        if board is not None and T_BinW is not None and have_t and cd and n >= 2:
            try:
                T_WB_inv = T_BinW.inverse()
                poses = [T_WB_inv.compose(values.atPose3(X(k))).compose(cal.T).matrix()
                         for k in keep]
                frames = sorted(cd.get("frames") or [], key=lambda f: f.timestamp)
                ft = np.array([f.timestamp for f in frames])
                tol = 0.5 * float(np.median(np.diff(ft))) if len(ft) > 1 else 1e-3
                want = np.asarray(t_states, float)[keep] - float(cal.td)
                times, near = match_frames(want, frames, tol)
                edges = np.searchsorted(ft, want, side="right")
                groups = [frames[(edges[j - 1] if j else 0):edges[j]]
                          for j in range(len(keep))]
                intr = np.asarray(cal.intr, float)
                cam = SimpleNamespace(camera_id=c, intrinsics=intr[:4],
                                      distortion=intr[4:], distortion_model=cal.model,
                                      image_size=tuple(cd.get("image_size") or (640, 480)))
                view["images"], _ = pose_images(poses, times, cam, board, frames,
                                                thumb_dir=cd.get("thumb_dir"))
                view["panel"] = pose_panel(poses, times, cam, board, frames,
                                           thumb_dir=cd.get("thumb_dir"),
                                           max_images=240 if len(cams) == 1 else 160,
                                           frame_groups=groups)
                view["ids"] = [[int(o.point_id) for o in f.observations]
                               if f is not None else [] for f in near]
            except Exception:
                pass                   # images are garnish; never fail the report
        views.append(view)
    return keep, views


def _build_3d_section(result, board, landmarks, cam_data=None, views_out=None, reference=None):
    """The calibration process and the solved rig as canvas views (`analysis.views`).

    Falls back to the older Plotly figures if the views can't be built, and
    returns "" when there is nothing to draw, so the report degrades
    gracefully (text + tables still render). `views_out` (a dict), when given,
    receives the two specs, for the `<report>_views.json` sidecar."""
    try:
        from mvis.analysis.views import assets_html, rig_html, rig_spec, scene_html, scene_spec
        sizes = {c: cd.get("image_size") for c, cd in enumerate(cam_data or []) if cd}
        parts = []
        rig = rig_spec(result, sizes, imu=_imu_block(result)[0], reference=reference)
        if rig:
            parts.append("<h3>Solved rig</h3>" + rig_html(rig))
        scene = None
        if getattr(result, "values", None) is not None:
            state_idx, views = _cam_views(result, board, cam_data)
            scene = scene_spec(result, board=board, views=views, state_idx=state_idx,
                               landmarks=landmarks, image_sizes=sizes, frames=True)
            if scene:
                parts.insert(0, "<h3>Calibration process</h3>" + scene_html(scene))
        if views_out is not None:
            views_out.update(scene=scene, rig=rig)
        return (assets_html() + "".join(parts)) if parts else ""
    except Exception:
        logging.getLogger(__name__).warning("report views failed; using the Plotly figures", exc_info=True)
    return _build_3d_section_plotly(result, board, landmarks, cam_data)


def _build_3d_section_plotly(result, board, landmarks, cam_data=None):
    """The older Plotly figures (scene + rig), kept as the fallback."""
    try:
        from mvis.analysis.report_shell import plotly_to_html
        from mvis.analysis.plot_3d import plot_cam_imu_scene, plot_cam_imu_rig
    except Exception:
        return ""
    parts = []
    try:
        if result.cameras:
            rig = plot_cam_imu_rig(result.cameras, getattr(result, "imus", None))
            parts.append("<h3>Sensor rig (extrinsics)</h3>" + plotly_to_html(rig))
    except Exception:
        pass
    try:
        values = getattr(result, "values", None)
        if values is not None:
            state_idx, views = _cam_views(result, board, cam_data)
            scene = plot_cam_imu_scene(
                values, board=board, T_BinW=getattr(result, "T_BinW", None),
                landmarks=landmarks, t_states=getattr(result, "t_states", None),
                state_idx=state_idx or None, views=views)
            parts.append("<h3>Trajectory &amp; scene</h3>" + plotly_to_html(scene))
    except Exception:
        pass
    return "".join(parts)



def _fig_html(fig):
    from mvis.analysis.report_shell import _fig_to_base64
    return ('<div class="plot-container">'
            f'<img src="data:image/png;base64,{_fig_to_base64(fig)}"></div>')


def _board_shape_body(rep, board):
    """Summary table and per-corner map of the estimated board shape (`diagnostics['board_heights']`),
    or "" when the run did not estimate it. The map needs the board; without it only the table shows."""
    if not rep or rep.get("heights_m") is None:
        return ""
    sig = rep.get("sigma_m")
    n_obs = rep.get("n_obs")
    seen = (np.asarray(n_obs) > 0) if n_obs is not None else None
    sig_med = (float(np.median(np.asarray(sig, float)[seen] if seen is not None else sig)) * 1e3
               if sig is not None and (seen is None or seen.any()) else None)
    rows = [["Peak to peak (mm)", f"{rep['p2p_mm']:.2f}" if rep.get("p2p_mm") is not None else "&mdash;"],
            ["RMS (mm)", f"{rep['rms_mm']:.2f}" if rep.get("rms_mm") is not None else "&mdash;"],
            ["Corners seen", f"{rep.get('n_seen', '&mdash;')} of {len(rep['heights_m'])}"],
            ["Median height σ (mm)", f"{sig_med:.2f}" if sig_med is not None else "&mdash;"]]
    body = ("<p>Per-corner height off the flat board, estimated with the camera-IMU solve "
            "(<code>estimate.board_points</code>). The mean plane and tilt belong to the board "
            "pose, so a flat board shows only noise here.</p>" + _tbl(["Metric", "Value"], rows))
    if board is not None:
        from mvis.analysis.plot_errors import plot_board_shape
        try:
            body += _fig_html(plot_board_shape(rep["heights_m"], board, n_obs))
        except ValueError as e:          # heights for a different board than the one passed
            logging.getLogger(__name__).warning("board shape map skipped: %s", e)
    return body


def _config_body(config):
    """Verbatim, copy-paste-ready run configuration.

    A `str` config is treated as raw YAML text and echoed as-is; a dict or
    (nested-)dataclass config is dumped to YAML mirroring the run-config
    schema, so the block can be saved back to a file and re-run."""
    if config is None:
        return ""
    if isinstance(config, str):
        text = config
    else:
        import dataclasses
        import yaml

        def clean(v):
            if dataclasses.is_dataclass(v) and not isinstance(v, type):
                return {f.name: clean(getattr(v, f.name))
                        for f in dataclasses.fields(v)}
            if isinstance(v, dict):
                return {k: clean(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [clean(x) for x in v]
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, (str, int, float, bool)) or v is None:
                return v
            return str(v)

        try:
            text = yaml.safe_dump(clean(config), default_flow_style=False,
                                  sort_keys=False)
        except Exception:
            return ""
    from mvis.analysis.report_shell import config_fold
    return config_fold(text)


def _results_body(result_files):
    """The written result files verbatim, one fold each (yaml / json)."""
    from mvis.analysis.report_shell import result_files_html
    return result_files_html(result_files)


def _mat_html(name, M, fmt="{:+.6f}"):
    """A labeled 3x3 matrix as a small table block."""
    rows = "".join(
        "<tr>" + "".join(f"<td>{fmt.format(float(v))}</td>" for v in row)
        + "</tr>" for row in M)
    return (f'<div style="display:inline-block;margin-right:28px;'
            f'vertical-align:top"><b>{name}</b><table>{rows}</table></div>')


def _dw_decomp(Dw) -> dict:
    """A calyx Dw split as Dw = R · U (`result_schema.calyx_dw_qr`): R the gyro misalignment
    rotation (matrix, log(R) in rad and its angle), U upper-triangular (per-axis scale on its
    diagonal, cross-coupling above it). The same split cam_imu.json's Dw_scale / Dw_shear /
    gyro_misalign_rotvec come from."""
    R, U = calyx_dw_qr(Dw)
    rv = np.asarray(log_so3(R), float)
    return dict(R=R.tolist(), U=U.tolist(), scale=np.diag(U).tolist(),
                shear=[float(U[0, 1]), float(U[0, 2]), float(U[1, 2])],
                rotvec_rad=rv.tolist(), angle_rad=float(np.linalg.norm(rv)))


def _dw_decomp_html(dd) -> str:
    return ("<div style=\"display:inline-block;vertical-align:top\"><b>Dw = R · U (calyx)</b>"
            + _tbl(["", "x", "y", "z"], [
                ["scale diag(U)", *[f"{v:+.6f}" for v in dd["scale"]]],
                ["shear U01 / U02 / U12", *[f"{v:+.6f}" for v in dd["shear"]]],
                ["misalignment log(R) (rad)", *[f"{v:+.6f}" for v in dd["rotvec_rad"]]]])
            + f"<p class='note'>Misalignment angle |log(R)| = {dd['angle_rad']:.6f} rad. R is the gyro "
            + "axes' rotation; U holds the per-axis scale and the cross-coupling.</p></div>"
            + _mat_html("R (misalignment)", dd["R"]) + _mat_html("U (scale + shear)", dd["U"]))


def _imu_intrinsics_html(entry):
    """Full IMU-intrinsic matrices (Dw / Da / Tg) for a base or aux entry; for a calyx IMU,
    Dw's R · U split under it."""
    if "Dw" not in entry:
        return ""
    return ("<h4>IMU intrinsics (full)</h4>"
            + _mat_html("Dw (gyro)", entry["Dw"])
            + (_dw_decomp_html(entry["Dw_decomp"]) if entry.get("Dw_decomp") else "")
            + _mat_html("Da (accel)", entry["Da"])
            + _mat_html("Tg (g-sensitivity)", entry.get("Tg", np.zeros((3, 3)))))


def _plot_reproj_frames(d):
    """Per-frame reprojection RMS timeline + histogram, from the diagnostics both
    pipeline modes record (feature: reproj_per_frame; board: reproj_rms_post_px),
    split per camera via reproj_frame_cam."""
    rms = d.get("reproj_per_frame", d.get("reproj_rms_post_px"))
    fcam = d.get("reproj_frame_cam")
    if rms is None or fcam is None:
        return None
    rms = np.asarray(rms, float).ravel()
    fcam = np.asarray(fcam, int).ravel()
    if rms.size == 0 or rms.shape != fcam.shape:
        return None
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 3.5), gridspec_kw={"width_ratios": [3, 1]})
    cams = np.unique(fcam)
    for c in cams:
        idx = np.flatnonzero(fcam == c)
        ax1.plot(idx, rms[idx], ".", ms=3, label=f"cam{c}")
    # guide lines = the headline (corner-pooled, Kalibr-convention) stats when
    # the diagnostics carry them, so plot and summary tiles agree; fall back to
    # per-frame stats for legacy results
    med_all = d.get("reproj_median_px")
    p95_all = d.get("reproj_p95_px")
    med_all = float(np.median(rms)) if med_all is None else float(med_all)
    p95_all = (float(np.percentile(rms, 95)) if p95_all is None
               else float(p95_all))
    ax1.axhline(med_all, color="k", lw=0.8, ls="--", label=f"median {med_all:.3f} px")
    ax1.axhline(p95_all, color="#B45309", lw=0.8, ls=":", label=f"p95 {p95_all:.3f} px")
    ax1.set_xlabel("frame (solve order)")
    ax1.set_ylabel("frame RMS (px)")
    ax1.legend(fontsize=8, ncol=min(len(cams) + 1, 5))
    ax2.hist([rms[fcam == c] for c in cams],
             bins=min(30, max(6, rms.size // 4)), stacked=True)
    ax2.set_xlabel("frame RMS (px)")
    ax2.set_ylabel("frames")
    fig.tight_layout()
    return fig


def _board_tags(board):
    """(corners per tag, number of tags, is_chessboard) as the camera-only report counts them."""
    if board is None:
        return 1, 0, True
    tags = getattr(board, "_num_tags", 0) or 0
    return (4, tags, False) if tags else (1, board.num_points(), True)


def _per_frame_robust(by_frame):
    """Per-frame median and P95 of the corner norms over frames 0..max (NaN where a frame has none)."""
    n = (max(by_frame) + 1) if by_frame else 0
    med, p95 = np.full(n, np.nan), np.full(n, np.nan)
    for f, v in by_frame.items():
        e = np.linalg.norm(np.asarray(v, float), axis=1)
        med[f], p95[f] = np.median(e), np.percentile(e, 95)
    return med.tolist(), p95.tolist()


def _reproj_cams(d):
    """(camera, its `reproj_corners_by_cam` entry, its corner norms) for cameras with corners."""
    out = []
    for c, cd in enumerate(d.get("reproj_corners_by_cam") or []):
        if cd.get("vectors_by_frame"):
            e = np.linalg.norm(np.vstack(list(cd["vectors_by_frame"].values())), axis=1)
            out.append((c, cd, e))
    return out


def _reproj_table_html(d, board) -> str:
    """Per-camera reprojection table (mean / median / P90 / P95 / max, tags, corners); "" without
    per-corner data. Shown in the Summary and at the top of Reprojection Analysis. Tags are the
    distinct (frame, tag) pairs `by_camera` counted from the corner ids (so corner subsampling
    and partly seen tags count right); the column is left out on a chessboard or without ids."""
    cams = _reproj_cams(d)
    if not cams:
        return ""
    show_tags = (not _board_tags(board)[2]
                 and all(cd.get("n_tags") is not None for _c, cd, _e in cams))
    cols = (["Camera", "Mean (px)", "Median (px)", "Std (px)", "P90 (px)", "P95 (px)", "Max (px)"]
            + (["Tags"] if show_tags else []) + ["Corners"])
    stats = [(c, cd, corner_stats(e)) for c, cd, e in cams]
    rows = [[cam_label(c), f"{s['mean']:.3f}", graded(f"{s['median']:.3f}", _grade(s['median'], *_MED_GRADE)),
             f"{s['std']:.3f}", f"{s['p90']:.3f}", graded(f"{s['p95']:.3f}", _grade(s['p95'], *_P95_GRADE)),
             f"{s['max']:.2f}"]
            + ([f"{cd['n_tags']:,}"] if show_tags else []) + [f"{s['n_corners']:,}"]
            for c, cd, s in stats]
    html = (_tbl(cols, rows)
            + "<p class='note'>Per-corner error norms at the solution, every corner the solve used (as Kalibr pools "
            + "them); guard-skipped corners left out"
            + ("; Tags = distinct tags seen per frame, summed over frames" if show_tags else "")
            + ". Std is the population std, as Kalibr prints it.</p>")
    return html


def _reproj_camera_sections(d, board) -> list[tuple[int, str]]:
    """Per camera, the camera-only report's error analysis as one block: headline tiles, the
    per-frame median/P95, error per tag, u/v histograms, error vectors, and one row of the
    Kalibr-style scatter (coloured by frame) beside the corner heatmap. Plots needing the board
    or corner ids are left out without them. [] for results without per-corner data."""
    from mvis.analysis.plot_errors import (
        plot_corner_heatmap, plot_error_histogram_uv, plot_error_per_tag_robust,
        plot_error_vectors_dual, plot_per_frame_robust, plot_reprojection_scatter)
    cpt, num_tags, is_chessboard = _board_tags(board)
    out = []
    for c, cd, e in _reproj_cams(d):
        by_frame = cd["vectors_by_frame"]
        ev = np.vstack(list(by_frame.values()))
        ce = cd.get("corner_errors") or {}
        med, p95 = float(np.median(e)), float(np.percentile(e, 95))
        tiles = ('<div class="stat-row">'
                 + _stat(f"{med:.3f} px", "Median", _grade(med, *_MED_GRADE))
                 + _stat(f"{p95:.3f} px", "P95", _grade(p95, *_P95_GRADE))
                 + _stat(f"{e.mean():.3f} px", "Mean")
                 + _stat(f"{len(by_frame):,}", "Frames")
                 + _stat(f"{len(e):,}", "Corners") + "</div>")
        parts = [tiles, "<h3>Per frame</h3>",
                 _fig_html(plot_per_frame_robust(*_per_frame_robust(by_frame), c))]
        if board is not None and ce:
            tag_errors = {}
            for pid, errs in ce.items():
                tag_errors.setdefault(int(pid) // cpt, []).extend(errs)
            parts += [f"<h3>Per {'corner' if is_chessboard else 'tag'}</h3>",
                      _fig_html(plot_error_per_tag_robust(tag_errors, num_tags, c,
                                                          is_chessboard=is_chessboard))]
        parts += ["<h3>Error distribution</h3>", _fig_html(plot_error_histogram_uv(ev, c)),
                  _fig_html(plot_error_vectors_dual(ev, by_frame, c))]
        row = [plot_reprojection_scatter(by_frame, c)]
        if board is not None and ce:
            row.append(plot_corner_heatmap(ce, board, c))
        imgs = "".join(_fig_html(f) for f in row)
        parts += ["<h3>Scatter by frame" + (" and error per board corner" if len(row) == 2 else "")
                  + "</h3>", f'<div class="plot-row">{imgs}</div>' if len(row) == 2 else imgs]
        out.append((c, "".join(parts)))
    return out


def _plot_excitation(result):
    """Body-frame angular velocity finite-differenced from the solved trajectory
    states X(k) — the excitation/observability context for lever + td."""
    values = getattr(result, "values", None)
    t = getattr(result, "t_states", None)
    if values is None or t is None or len(t) < 3:
        return None
    try:
        import gtsam
        X = gtsam.symbol_shorthand.X
        R = []
        for k in range(len(t)):
            if not values.exists(X(k)):
                return None
            R.append(values.atPose3(X(k)).rotation().matrix())
    except Exception:
        return None
    t = np.asarray(t, float)
    dt = np.maximum(np.diff(t), 1e-9)
    w = np.stack([log_so3(R[k].T @ R[k + 1]) / dt[k] for k in range(len(t) - 1)])
    wd = np.rad2deg(w)
    tm = 0.5 * (t[:-1] + t[1:]) - t[0]
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4.5), sharex=True)
    ax1.plot(tm, np.linalg.norm(wd, axis=1), lw=1.0)
    ax1.set_ylabel("|\u03c9| (\u00b0/s)")
    ax1.set_title("Angular velocity (body frame, finite-differenced states)",
                  fontsize=10)
    for i, lbl in enumerate("xyz"):
        ax2.plot(tm, wd[:, i], lw=0.9, label=f"\u03c9{lbl}")
    ax2.set_ylabel("\u03c9 (\u00b0/s)")
    ax2.set_xlabel("t (s)")
    ax2.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    return fig


def _plot_bias_evolution(result):
    """Per-state gyro / accel bias over the run, each axis minus its own mean (the mean is in the
    legend): the solver keeps one random-walk bias per state
    (_Vk(k) = [v, b_g, b_a] in result.values). Without values, falls back to the first-to-last
    chord b -> b + drift, titled as such (not the solver's model)."""
    t = getattr(result, "t_states", None)
    if t is None or len(t) < 2:
        return None
    t = np.asarray(t, float)
    dur = float(t[-1] - t[0])
    if dur <= 0:
        return None
    series, note = None, ""
    values = getattr(result, "values", None)
    if values is not None:
        try:
            import gtsam
            vb = np.array([np.asarray(values.atVector(gtsam.symbol('v', k)), float)
                           for k in range(len(t))])
            series = {"gyro": (t - t[0], vb[:, 3:6]), "accel": (t - t[0], vb[:, 6:9])}
        except Exception:       # values without per-state _Vk keys
            series = None
    if series is None:
        tt = np.array([0.0, dur])
        series = {name: (tt, np.array([b, np.asarray(b, float) + np.asarray(d, float)], float))
                  for name, b, d in (("gyro", result.b_g, result.b_g_drift),
                                     ("accel", result.b_a, result.b_a_drift))}
        note = ", first-to-last chord"
    import matplotlib.pyplot as plt
    fig, (axg, axa) = plt.subplots(1, 2, figsize=(12, 3))
    for ax, name, unit in ((axg, "gyro", "rad/s"), (axa, "accel", "m/s\u00b2")):
        tt, B = series[name]
        mean = B.mean(axis=0)       # axes sit far apart: centre each so its drift is visible
        for i, lbl in enumerate("xyz"):
            ax.plot(tt, B[:, i] - mean[i], marker="." if len(tt) <= 50 else None,
                    label=f"{lbl} (mean {mean[i]:+.4g})")
        ax.set_title(f"{name} bias minus axis mean ({unit}{note})", fontsize=10)
        ax.set_xlabel("t (s)")
        ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    return fig


_IMU_BLOCK_INFO = {          # label, raw unit, raw format
    "theta": ("rotation θ", "rad", "{:.2e}"), "p": ("position p", "m", "{:.2e}"),
    "v": ("velocity v", "m/s", "{:.2e}"), "bg": ("gyro bias step Δb_g", "rad/s", "{:.2e}"),
    "ba": ("accel bias step Δb_a", "m/s²", "{:.2e}")}
_CHI2_3_95 = 2.7955          # sqrt of the chi-square(3) 95 % quantile: |W r_b| above it is unusual


def _imu_residual_body(d):
    """The IMU preintegration residual per state block — rotation, position, velocity, gyro and accel
    bias steps — over the run (one row per block, whitened |W r_b| per interval) and a table of each
    block's raw residual in its unit, its normalized residual and its share of the IMU cost. Falls back
    to the whole-state per-interval cost for results recorded before the per-block diagnostic."""
    parts = []
    blk = d.get("imu_block_residuals")
    split = d.get("imu_cost_split") if isinstance(d.get("imu_cost_split"), dict) else {}
    tot = sum(split.values()) or 1.0
    if isinstance(blk, dict) and blk.get("white") is not None and len(blk["white"]):
        import matplotlib.pyplot as plt
        names = list(blk["blocks"])
        W = np.asarray(blk["white"], float)
        R = np.asarray(blk["raw"], float)
        dt = np.asarray(blk.get("dt"), float)
        t = (np.concatenate([[0.0], np.cumsum(dt[:-1])]) if np.all(np.isfinite(dt)) else np.arange(len(W)))
        xlab = "time from the first state (s)" if np.all(np.isfinite(dt)) else "state interval"
        fig, axes = plt.subplots(len(names), 1, figsize=(12, 1.55 * len(names) + 0.6), sharex=True)
        for ax, j, name in zip(np.atleast_1d(axes), range(len(names)), names):
            ax.plot(t, W[:, j], lw=0.9, color="C0")
            ax.axhline(_CHI2_3_95, color="#DC2626", ls="--", lw=0.8)
            ax.axhline(float(np.nanmedian(W[:, j])), color="#FF9800", ls=":", lw=0.9)
            ax.set_ylabel(_IMU_BLOCK_INFO.get(name, (name,))[0].split(" ")[0] + "\n|W r|", fontsize=9)
            ax.set_ylim(bottom=0)
        np.atleast_1d(axes)[-1].set_xlabel(xlab)
        np.atleast_1d(axes)[0].set_title("Normalized IMU residual per state block (dashed: χ²₃ 95 %, dotted: median)",
                                         loc="left")
        fig.tight_layout()
        parts.append("<h3>Per state block over the run</h3>" + _fig_html(fig))
        rows = []
        for j, name in enumerate(names):
            label, unit, fmt = _IMU_BLOCK_INFO.get(name, (name, "", "{:.3g}"))
            r = R[:, j][np.isfinite(R[:, j])]
            w = W[:, j][np.isfinite(W[:, j])]
            raw_cells = ([fmt.format(np.median(r)), fmt.format(np.percentile(r, 95)), fmt.format(r.max())]
                         if r.size else ["—"] * 3)
            rows.append([label, unit, *raw_cells,
                         f"{np.median(w):.2f}", f"{np.percentile(w, 95):.2f}",
                         f"{100 * np.mean(w > _CHI2_3_95):.1f}%",
                         f"{100 * split.get(name, 0.0) / tot:.1f}%" if split else "—"])
        parts.append("<h3>Per state block</h3>" + _tbl(
            ["Block", "Unit", "Raw median", "Raw P95", "Raw max", "|W r| median", "|W r| P95", "Over χ²₃ 95 %",
             "Cost share"], rows)
            + "<p class='note'>Raw: the block's residual per interval in its unit (rotation and velocity "
              "against the integrated gyro and accelerometer, the bias steps between states). |W r|: the same "
              "whitened by the solve's IMU noise; with the noise right it is about √3 ≈ 1.7, and about 5 % of "
              "intervals exceed the χ²₃ 95 % line.</p>")
        return "".join(parts)
    ic = d.get("imu_interval_cost")                 # older results: the whole IMU state only
    if ic is not None and len(ic):
        import matplotlib.pyplot as plt
        ic = np.asarray(ic, float).ravel()
        fig, ax = plt.subplots(figsize=(12, 2.8))
        ax.bar(np.arange(ic.size), ic, width=0.9)
        ax.set_xlabel("state interval")
        ax.set_ylabel("IMU cost")
        fig.tight_layout()
        parts.append("<h3>Per-interval cost</h3>" + _fig_html(fig))
    if split:
        parts.append("<h3>Cost split</h3>" + _tbl(
            ["residual rows", "cost", "share"],
            [[k, f"{v:.1f}", f"{100 * v / tot:.1f}%"] for k, v in split.items()]))
    return "".join(parts)


def _timing_body(d):
    tm = d.get("timings")
    if not isinstance(tm, dict) or not tm:
        return ""
    vals = {k: float(v) for k, v in tm.items() if isinstance(v, (int, float))}
    if not vals:
        return ""
    tot = sum(vals.values())
    rows = [[k, f"{v:.2f}", f"{100 * v / tot:.1f}%" if tot else "&mdash;"]
            for k, v in vals.items()]
    rows.append(["<b>total</b>", f"<b>{tot:.2f}</b>", ""])
    return _tbl(["stage", "s", "share"], rows)


@report_plot_style
def cam_imu_html_report(result, reference=None, title="Camera-IMU Calibration",
                        board=None, landmarks=None, config=None,
                        cam_data=None, result_files=None, views_out=None) -> str:
    """Self-contained HTML cam-IMU report on the shared report shell.

    `board` / `landmarks` are optional and feed the 3D geometry section.
    `config` (dict or object) is optionally echoed as a Configuration section.
    Data-optional sections (config echo, per-frame reprojection, excitation,
    bias evolution, IMU residuals, timing) render only when the result carries
    the corresponding diagnostics/state, so old results still render."""
    rows = _cam_rows(result)
    base, aux = _imu_block(result)
    d = result.diagnostics
    sections, toc, idx = [], [], [0]

    def sec(sec_id, label, body):
        idx[0] += 1
        toc.append((sec_id, f"{idx[0]}. {label}"))
        alt = " section-alt" if idx[0] % 2 == 0 else ""
        return f'<div class="section{alt}" id="{sec_id}"><h2>{idx[0]}. {label}</h2>{body}</div>'

    # ---- 1. Summary -------------------------------------------------------
    reproj = float(d.get("reproj_rms_px", float("nan")))
    med = d.get("reproj_median_px")
    p95 = d.get("reproj_p95_px")
    cam_chi2 = float(d.get("cam_chi2_dof", float("nan")))
    imu_chi2 = float(d.get("imu_chi2_dof", float("nan")))
    status_badge = ('<span class="badge badge-good">&#10003; OK</span>' if result.ok
                    else '<span class="badge badge-poor">&#10007; FAILED</span>')
    # each number once: the tiles carry the fit with its grade; a one-line strip the run's facts
    stat_html = (
        '<div class="stat-row">'
        + _stat(status_badge, "Status")
        + (_stat(f"{med:.3f} px", "Reproj median", _grade(med, *_MED_GRADE))
           + _stat(f"{p95:.3f} px", "Reproj P95", _grade(p95, *_P95_GRADE))
           if med is not None and p95 is not None else
           _stat(f"{reproj:.3f} px", "Reproj RMS", _grade(reproj, *_RMS_GRADE)))
        + _stat(f"{cam_chi2:.3f}", "Camera <span class='nt'>χ²/dof</span>", _grade(cam_chi2, *_CHI2_GRADE))
        + _stat(f"{imu_chi2:.3f}", "IMU <span class='nt'>χ²/dof</span>", _grade(imu_chi2, *_CHI2_GRADE))
        + _stat(len(rows), "Cameras")
        + _stat(1 + len(aux), "IMUs")
        + _stat(d.get("iterations", 0), "Iterations")
        + "</div>")
    facts = [("Trajectory states", f"{d.get('n_states', 0):,}"),
             ("IMU model", getattr(result, "imu_model", "—"))]
    if d.get("gravity") is not None:      # |g| held fixed: the accel scale absorbs g_true / g_used
        facts.append(("Gravity", html.escape(describe_gravity(float(d["gravity"]), d.get("gravity_source", "default"),
                                                             d.get("gravity_site"), d.get("gravity_place")))))
    if d.get("t_ga") is not None:
        tga_s = d.get("t_ga_sigma")
        facts.append(("t_ga gyro&rarr;accel", f"{d['t_ga'] * 1e3:+.3f}"
                      + (f" &plusmn; {tga_s * 1e3:.3f}" if tga_s else "") + " ms"))
    diag_tbl = ("<div class='facts'>" + "".join(f"<span><b>{k}</b> {v}</span>" for k, v in facts) + "</div>")
    cam_table = _reproj_table_html(d, board)     # Summary and Reprojection Analysis
    sections.append(sec("sec-summary", "Summary", stat_html + diag_tbl
                        + (f"<h3>Reprojection per camera</h3>{cam_table}" if cam_table else "")
                        + _param_folds_html(rows, base, aux, d, config, result_files)))

    # ---- 2. 3D geometry ---------------------------------------------------
    body_3d = _build_3d_section(result, board, landmarks, cam_data, views_out, reference)
    if body_3d:
        sections.append(sec("sec-3d", "3D Geometry", body_3d))

    # ---- 3. Cameras -------------------------------------------------------
    crows = []
    for r in rows:
        crows.append([
            cam_label(r["id"]), r["model"],
            (f"{r['reproj_mean_px']:.3f} / {r['reproj_median_px']:.3f} / {r['reproj_std_px']:.3f}"
             f" / {r['reproj_p95_px']:.2f}")
            if all(r.get(k) is not None for k in ("reproj_mean_px", "reproj_std_px", "reproj_median_px", "reproj_p95_px"))
            else f"{r['reproj_median_px']:.3f} / {r['reproj_p95_px']:.2f}"
            if r.get("reproj_median_px") is not None else f"{r['reproj_rms_px']:.3f}",
            f"{r['fx']:.1f}, {r['fy']:.1f}, {r['cx']:.1f}, {r['cy']:.1f}",
            f"{r['td_ms']:+.3f}",
            f"{r['tr_ms']:.3f}{' (pin)' if r['tr_pinned'] else ''}",
            f"{r['lever_norm_mm']:.2f}", f"{r['rot_angle_rad']:.5f}",
            f"{r['baseline_norm_mm']:.2f}" if "baseline_norm_mm" in r else "—",
        ])
    reproj_head = ("reproj mean/med/std/p95 px" if all(r.get("reproj_mean_px") is not None and r.get("reproj_std_px")
                                                       is not None for r in rows)
                   else "reproj med/p95 px")
    cam_body = _tbl(["cam", "model", reproj_head, "fx,fy,cx,cy", "td ms", "tr ms",
                     "lever |mm|", "rot |rad|", "baseline mm"], crows)
    # per-camera detail (T_C->I lever / rotation with sigma, distortion)
    for r in rows:
        det = _tbl([f"cam{r['id']} — T_C&rarr;I", "x", "y", "z", "|·|"], [
            ["lever (mm)", *[f"{v:.2f}" for v in r["lever_mm"]], f"{r['lever_norm_mm']:.2f}"],
            ["lever σ (mm)", *[f"{v:.2f}" for v in r["lever_sig_mm"]], ""],
            ["rotation (rad)", *[f"{v:.5f}" for v in r["rot_rad"]], f"{r['rot_angle_rad']:.5f}"],
            ["rotation σ (rad)", *[f"{v:.2e}" for v in r["rot_sig_rad"]], ""],
        ])
        det += _tbl([f"cam{r['id']} — temporal / distortion", "value"], [
            ["td (ms)", f"{r['td_ms']:+.3f}  (σ {r['td_sig_ms']:.3f})"],
            ["tr (ms)", f"{r['tr_ms']:.3f}{' [pinned]' if r['tr_pinned'] else ''}"],
            ["intr σ (fx,fy,cx,cy)",
             " ".join(f"{x:.3f}" for x in r["intr_sig"])
             if r.get("intr_sig") else "&mdash;"],
            [f"distortion{_dist_label(r)}", _dist_values(r)],
            ["dist σ", " ".join(f"{x:.2e}" for x in r["dist_sig"])
             if r.get("dist_sig") else "&mdash;"],
            *([["valid domain", _kb_domain_text(r)]] if _kb_domain_text(r) else []),
            *([["corner accounting", _corner_accounting_html(r)]]
              if r.get("corner_accounting") else []),
        ])
        cam_body += f"<h3>cam{r['id']}</h3>" + det
    sections.append(sec("sec-cameras", "Cameras", cam_body))

    # ---- Board shape (only when estimate.board_points was on) ------------
    shape_body = _board_shape_body(d.get("board_heights"), board)
    if shape_body:
        sections.append(sec("sec-board-shape", "Board Shape", shape_body))

    # ---- Reprojection analysis: all cameras, then one subsection per camera
    fig = _plot_reproj_frames(d)
    if fig is not None or cam_table:
        per_cam = _reproj_camera_sections(d, board)
        panels = "".join(f'<div class="cam-panel" id="sec-reproj-cam{c}" data-label="cam{c}">'
                         f"<h3>{cam_label(c)} — reprojection</h3>{body}</div>" for c, body in per_cam)
        sections.append(sec("sec-reproj", "Reprojection Analysis",
                            (_fig_html(fig) if fig is not None else "")
                            + (f"<h3>Per camera</h3>{cam_table}" if cam_table else "")
                            + (f"<div class='cam-tabs'>{panels}</div>" if panels else "")))
        n = idx[0]
        for k, (c, _body) in enumerate(per_cam, 1):    # TOC entries open the camera's tab
            toc.append((f"sec-reproj-cam{c}", f"&nbsp;&nbsp;&nbsp;{n}.{k} cam{c}"))

    # ---- Motion & excitation ----------------------------------------------
    fig = _plot_excitation(result)
    if fig is not None:
        sections.append(sec(
            "sec-motion", "Motion &amp; Excitation",
            _fig_html(fig)
            + "<p>Body-frame angular velocity finite-differenced from the "
              "solved trajectory states — the excitation context for "
              "lever-arm / td observability.</p>"))

    # ---- 3. Base IMU ------------------------------------------------------
    irows = [["b_g", " ".join(f"{x:.5f}" for x in base["b_g"])],
             ["b_g drift", " ".join(f"{x:.5f}" for x in base["b_g_drift"])],
             ["b_a", " ".join(f"{x:.4f}" for x in base["b_a"])],
             ["b_a drift", " ".join(f"{x:.4f}" for x in base["b_a_drift"])]]
    if base["pinned"]:
        irows.append(["Dw/Da", "pinned (phase A)"])
    elif "R_wtoI_rad" in base:
        irows.append(["R_wtoI (rad)", " ".join(f"{x:.5f}" for x in base["R_wtoI_rad"])])
    if base.get("t_ga") is not None:
        tga_s = base.get("t_ga_sigma")
        irows.append(["t_ga gyro&rarr;accel (ms)",
                      f"{base['t_ga'] * 1e3:+.3f}"
                      + (f" &plusmn; {tga_s * 1e3:.3f}" if tga_s else "")
                      + "  <i>(gyro clock is reference: accel stamped t sampled"
                        " at t &minus; t_ga)</i>"])
    imu_body = _tbl(["param", "value"], irows)
    imu_body += _imu_intrinsics_html(base)      # full Dw / Da / Tg matrices
    fig = _plot_bias_evolution(result)
    if fig is not None:
        imu_body += "<h3>Bias evolution</h3>" + _fig_html(fig)
    sections.append(sec("sec-imu", f"Base IMU ({base['name']}, model {base['imu_model']})",
                        imu_body))

    # ---- 4. Aux IMUs ------------------------------------------------------
    if aux:
        aux_body = ""
        for a in aux:
            arows = [["td vs base (ms)", f"{a['td_ms']:+.3f}"]]
            if a.get("t_ga") is not None:
                tga_s = a.get("t_ga_sigma")
                arows.append(["t_ga gyro&rarr;accel (ms)",
                              f"{a['t_ga'] * 1e3:+.3f}"
                              + (f" &plusmn; {tga_s * 1e3:.3f}" if tga_s else "")])
            if "chi2_dof" in a:
                arows.append(["χ²/dof", f"{a['chi2_dof']:.2f}"])
            if "p_IsinI_mm" in a:
                arows.append(["T_IstoI lever p_IsinI (mm)",
                              "[" + ", ".join(f"{v:.2f}" for v in a["p_IsinI_mm"])
                              + f"]  |{np.linalg.norm(a['p_IsinI_mm']):.2f}|"])
            if "rot_rad" in a:
                arows.append(["T_IstoI rotation (rad)",
                              "[" + ", ".join(f"{v:.5f}" for v in a["rot_rad"])
                              + f"]  |{a['rot_angle_rad']:.5f}|"])
            if "b_g" in a:
                arows.append(["b_g", " ".join(f"{x:.5f}" for x in a["b_g"])])
            if "b_a" in a:
                arows.append(["b_a", " ".join(f"{x:.4f}" for x in a["b_a"])])
            if "Dw" not in a:
                arows.append(["Dw/Da", "pinned"])
            aux_body += (f"<h3>{a['name']}</h3>"
                         + _tbl(["param", "value"], arows)
                         + _imu_intrinsics_html(a))
        sections.append(sec(
            "sec-aux", "Aux IMUs",
            "<p>Per-IMU parameters in the base-IMU format; the spatial"
            " (T_IstoI) and temporal (td) rows are relative to the base"
            " IMU.</p>" + aux_body))

    # ---- IMU residuals ----------------------------------------------------
    resid_body = _imu_residual_body(d)
    if resid_body:
        sections.append(sec("sec-imu-resid", "IMU Residuals", resid_body))

    # ---- 5. vs reference --------------------------------------------------
    deltas = _ref_deltas(rows, reference)
    if deltas:
        drows = []
        for c in deltas["cams"]:
            lev = c.get("lever_delta_mm")
            rot = c.get("rotation_delta_rad")
            drows.append([
                f"cam{c['id']}",
                f"{lev:.2f}" if lev is not None else "—", _badge(_grade(lev, 5.0, 15.0)),
                f"{rot:.5f}" if rot is not None else "—",
                _badge(_grade(rot, *_ROT_DELTA_GRADE_RAD)),
            ])
        body = _tbl(["", "lever Δ mm", "", "rotation Δ rad", ""], drows)
        if deltas["baseline"]:
            b = deltas["baseline"]
            body += (f"<p>baseline {b['got_mm']:.2f} mm vs {b['ref_mm']:.2f} mm "
                     f"(Δ <b>{b['delta_mm']:.2f} mm</b>) "
                     f"{_badge(_grade(b['delta_mm'], 5.0, 15.0))}</p>")
        sections.append(sec("sec-reference", f"vs reference: {deltas['label']}", body))

    # ---- Appendix: timing, result files, configuration (closed) ---------------
    from mvis.analysis.report_shell import folded
    parts, letter = [], iter("ABCDEF")
    timing_body = _timing_body(d)
    if timing_body:
        parts.append(f'<div id="sec-timing"><h3>{next(letter)}. Timing</h3>'
                     + folded("Solve stage timings (click to expand)", timing_body) + "</div>")
    res_body = _results_body(result_files)
    if res_body:
        parts.append(f'<div id="sec-results"><h3>{next(letter)}. Result files</h3>{res_body}</div>')
    cfg_body = _config_body(config)
    if cfg_body:
        parts.append(f'<div id="sec-config"><h3>{next(letter)}. Configuration</h3>{cfg_body}</div>')
    if parts:
        sections.append(sec("sec-appendix", "Appendix", "".join(parts)))

    return render_page(title, toc, sections)


def write_cam_imu_report(result, html_path=None, text_path=None, reference=None,
                         board=None, landmarks=None, config=None, cam_data=None,
                         result_files=None):
    """Write the cam-IMU report (HTML and/or text). Returns the text report.

    `board` (+ `landmarks` for feature mode) feed the HTML 3D scene plot; both are
    optional — the report renders fine without them (rig plot still draws)."""
    txt = cam_imu_text_report(result, reference)
    if text_path:
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(txt + "\n")
    if html_path:
        views = {}
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(cam_imu_html_report(result, reference, board=board,
                                        landmarks=landmarks, config=config,
                                        cam_data=cam_data,
                                        result_files=result_files, views_out=views))
        if views.get("scene") or views.get("rig"):
            write_views_sidecar(views, views_sidecar_path(html_path))
    return txt


def views_sidecar_path(html_path) -> str:
    """`<report stem>_views.json` beside the report."""
    import os
    stem = os.path.splitext(str(html_path))[0]
    stem = stem[:-len("_report")] if stem.endswith("_report") else stem
    return stem + "_views.json"


def write_views_sidecar(views: dict, path) -> None:
    """The report's two view specs without any camera frame, for openmvis.com."""
    import json
    from mvis.analysis.views.spec import without_frames
    out = {"scene": without_frames(views.get("scene")), "rig": views.get("rig")}
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"), allow_nan=False)
