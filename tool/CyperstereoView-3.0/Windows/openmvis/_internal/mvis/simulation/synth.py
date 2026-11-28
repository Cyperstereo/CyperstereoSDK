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
"""Synthetic cams and IMUs from a `GroundTruth`, the recovery solve, and its z-score table.

`build_synth` turns a GT (real or made up) into the `cams`/IMU inputs of
`run_cam_imu_calibration_multi`; `zscore_table` scores a solve against that GT. Used by
`mvis.simulation.run` and the recovery tests.
"""
import json
import os
import sys

import numpy as np
import gtsam

from mvis.calibration.pipeline_cam_imu import run_cam_imu_calibration_multi
from mvis.core.camera_models import DIST_NAMES, canonical_cal_vec
from mvis.core.config_parser import parse_cam_imu_config
from mvis.detection.board_factory import create_board
from mvis.imu.imu_model import ImuModel, parse_imu_model
from mvis.simulation.camera_measurements import simulate_camera
from mvis.simulation.from_result import cal_from_vec, fit_bspline_from_states, gt_cam_model, gt_from_json
from mvis.simulation.imu_measurements import simulate_imu

STATE_DT = 1.0 / 30.0


def _imu_intr(xi24, model_name):
    return ImuModel.from_tangent(np.asarray(xi24, float), parse_imu_model(model_name))


def _spline(gt):
    return fit_bspline_from_states(np.asarray(gt.t_states, float),
                                   [np.asarray(T, float) for T in gt.T_states],
                                   STATE_DT)


def build_synth(gt, frame_times_per_cam, imu_times_per_imu, sigma_px, imu_sigmas,
                seed, pin_intrinsics=False):
    """Generate synthetic cams/aux_imus dicts from GT. intrinsics FREE by default
    (pin_intrinsics=True reproduces the pinned-prior recovery for the
    focal<->depth-gauge verification)."""
    spline, _err = _spline(gt)
    Tb = np.asarray(gt.T_BinW, float)
    if gt.board_points_local is None:
        raise ValueError("build_synth needs gt.board_points_local (the board's local points)")
    pts_local = np.asarray(gt.board_points_local, float)
    pts_world = (Tb[:3, :3] @ pts_local.T).T + Tb[:3, 3]
    g_inW = np.asarray(gt.g_inW, float)

    cams = []
    for ci, cgt in enumerate(gt.cams):
        model = gt_cam_model(cgt)     # GT json 'model' (legacy: from is_fisheye)
        cal = cal_from_vec(cgt["intr8"], cgt["is_fisheye"], model)
        T_CtoI = gtsam.Pose3(np.asarray(cgt["T_CtoI"], float))
        cs = simulate_camera(spline, np.asarray(frame_times_per_cam[ci], float),
                             cal, T_CtoI, pts_world, tuple(cgt["image_size"]),
                             td=cgt["td"], sigma_px=sigma_px, seed=seed + ci)
        cams.append(dict(
            t_cam=cs.t, ids_list=cs.ids, uv_list=cs.uv,
            cal_vec=np.asarray(cgt["intr8"], float),
            is_fisheye=cgt["is_fisheye"],
            model=model,
            image_size=tuple(cgt["image_size"]),
            pin_intrinsics=pin_intrinsics,        # FREE intrinsics by default
            estimate_readout=False, rolling_mode=0))

    base = gt.imus[0]
    bs = simulate_imu(spline, np.asarray(imu_times_per_imu[0], float),
                      _imu_intr(base["xi24"], gt.imu_model), g_inG=g_inW,
                      b_g0=gt.b_g, b_a0=gt.b_a, seed=seed + 100, **imu_sigmas)
    t_imu, w_m, a_m = bs.t, bs.w_m, bs.a_m

    aux_imus = []
    for si, agt in enumerate(gt.imus[1:], start=1):
        T = np.asarray(agt["T_IstoI"], float)
        s = simulate_imu(spline, np.asarray(imu_times_per_imu[si], float),
                         _imu_intr(agt["xi24"], gt.imu_model), g_inG=g_inW,
                         R_IstoI=T[:3, :3], p_IsinI=T[:3, 3], td=agt["td"],
                         b_g0=gt.b_g, b_a0=gt.b_a, seed=seed + 200 + si, **imu_sigmas)
        aux_imus.append(dict(t_imu=s.t, w_m=s.w_m, a_m=s.a_m, name=agt["name"]))
    return cams, t_imu, w_m, a_m, aux_imus


def recover(cams, board_points, t_imu, w_m, a_m, aux_imus, config):
    return run_cam_imu_calibration_multi(cams, board_points, t_imu, w_m, a_m,
                                         aux_imus=aux_imus or None, config=config)


def intr_row_names(model) -> list[str]:
    """Intrinsic row names of a camera model, as `zscore_table` names its rows: fx, fy, cx, cy,
    then k1, k2, p1, p2 for radtan/equidistant (the historical names) or the registry
    `DIST_NAMES` for the other models."""
    return ["fx", "fy", "cx", "cy"] + (
        ["k1", "k2", "p1", "p2"] if model in ("radtan", "equidistant")
        else list(DIST_NAMES[model]))


def zscore_table(gt, result):
    """One row per checked param: GT vs recovered, error/sigma. calib_sigma is the
    (8 + n_intr)-vec on the base xi [rot(3), pos(3), td, tr, intr]; aux calib_sigma(31) is
    [pose(6), td, intr24]."""
    rows = []

    def row(name, g, e, s):
        s = float(s) if (s is not None and np.isfinite(s) and s > 0) else float("nan")
        z = float((e - g) / s) if np.isfinite(s) else float("nan")
        rows.append(dict(param=name, gt=float(g), est=float(e), err=float(e - g),
                         sigma=s, z=z, pass_=bool(np.isfinite(z) and abs(z) < 3.0)))

    for ci, (cgt, e) in enumerate(zip(gt.cams, result.cameras)):
        cal = e["calib"]
        intr_names = intr_row_names(cal.model)
        n_sig = 8 + len(intr_names)          # [rot3, pos3, td, tr, intr(n_intr)]
        sig_raw = e.get("calib_sigma")
        sig = np.asarray(sig_raw, float) if sig_raw is not None else np.full(n_sig, float("nan"))
        # Guard: ensure at least n_sig elements (fall back to NaN for out-of-range)
        if len(sig) < n_sig:
            sig = np.concatenate([sig, np.full(n_sig - len(sig), float("nan"))])
        # extrinsics: GT T_CtoI vs recovered cal.T (compare in tangent)
        Tg = gtsam.Pose3(np.asarray(cgt["T_CtoI"], float))
        d = gtsam.Pose3.Logmap(Tg.inverse().compose(cal.T))          # 6-vec
        for a in range(3):
            row(f"cam{ci}/rot{a}", 0.0, d[a], sig[a])
        for a in range(3):
            row(f"cam{ci}/pos{a}", 0.0, d[3 + a], sig[3 + a])
        row(f"cam{ci}/td", cgt["td"], cal.td, sig[6])
        # legacy short GT (8-long omni: no p2) zero-padded to the model length
        g_intr = canonical_cal_vec(cal.model, cgt["intr8"])
        for a, nm in enumerate(intr_names):
            idx = 8 + a
            s_val = sig[idx] if idx < len(sig) else float("nan")
            row(f"cam{ci}/{nm}", g_intr[a], float(cal.intr[a]), s_val)

    # IMU intrinsics (Dw/Da diag) — base + aux
    gt_by = {m["name"]: m for m in gt.imus}
    base_ii_gt = _imu_intr(gt.imus[0]["xi24"], gt.imu_model)
    base_ii_est = ImuModel.from_tangent(np.asarray(result.imu_intr_xi, float),
                                        parse_imu_model(gt.imu_model))
    isig_raw = getattr(result, "imu_intr_sigma", None)
    isig = (np.asarray(isig_raw, float) if isig_raw is not None else None)
    for a in range(3):
        s_val = float(isig[a]) if isig is not None and a < len(isig) else float("nan")
        row(f"base/Dw{a}", base_ii_gt.Dw()[a, a], base_ii_est.Dw()[a, a], s_val)
    for m in (result.imus or []):
        g_entry = gt_by.get(m.get("name"))
        if g_entry is None:
            continue
        xs = np.asarray(m["intr_xi24"], float)
        ii_e = ImuModel.from_tangent(xs, parse_imu_model(gt.imu_model))
        ii_g = _imu_intr(g_entry["xi24"], gt.imu_model)
        sig_raw_a = m.get("calib_sigma")
        sig_a = np.asarray(sig_raw_a, float) if sig_raw_a is not None else None
        # T_IstoI in result.imus is a gtsam.Pose3
        T_IstoI_est = m["T_IstoI"]
        if isinstance(T_IstoI_est, np.ndarray):
            T_IstoI_est = gtsam.Pose3(T_IstoI_est)
        Tg_imu = gtsam.Pose3(np.asarray(g_entry["T_IstoI"], float))
        d = gtsam.Pose3.Logmap(Tg_imu.inverse().compose(T_IstoI_est))
        for a in range(6):
            s_val = float(sig_a[a]) if sig_a is not None and a < len(sig_a) else float("nan")
            row(f"{g_entry['name']}/pose{a}", 0.0, d[a], s_val)
        s_td = float(sig_a[6]) if sig_a is not None and 6 < len(sig_a) else float("nan")
        row(f"{g_entry['name']}/td", g_entry["td"], float(m["td"]), s_td)
        for a in range(3):
            s_val = float(sig_a[7 + a]) if sig_a is not None and 7 + a < len(sig_a) else float("nan")
            row(f"{g_entry['name']}/Dw{a}", ii_g.Dw()[a, a], ii_e.Dw()[a, a], s_val)
    return rows


def run_subset(gt_path, cfg_path, out_path, *, n_imus, seed=0, sigma_px=1.0, aux_imu_stride=None, on_synth=None):
    """Closed-loop recovery on a subset of a cached GT: cam0 + the first `n_imus` IMUs, same motion
    and noise densities as the full study. Writes the z-score rows to `out_path` and returns
    (gt, out). `on_synth(cfg, cams, t_imu, aux)` runs just before the solve (a script's progress line)."""
    gt = gt_from_json(gt_path)
    gt.cams = [gt.cams[0]]
    gt.imus = gt.imus[:n_imus]

    cfg = parse_cam_imu_config(cfg_path)
    if aux_imu_stride is not None:
        cfg.solve.aux_imu_stride = aux_imu_stride
    board = create_board(cfg.board)
    gt.board_points_local = np.asarray(board.all_points_3d(), float)

    frame_times = [np.asarray(gt.cams[0]["frame_times"], float)]
    imu_times = [np.asarray(m["sample_times"], float) for m in gt.imus]
    sig = dict(sigma_w=0.01, sigma_a=0.02, sigma_bg=0.001, sigma_ba=0.002)

    cams, t_imu, w_m, a_m, aux = build_synth(gt, frame_times, imu_times, sigma_px, sig, seed)
    if on_synth is not None:
        on_synth(cfg, cams, t_imu, aux)
    res = recover(cams, gt.board_points_local, t_imu, w_m, a_m, aux, cfg.solve)
    rows = zscore_table(gt, res)
    out = dict(seed=seed, sigma_px=sigma_px,
               reproj_rms_px=float(res.diagnostics["reproj_rms_px"]),
               rows=rows, n_fail=sum(1 for r in rows if not r["pass_"]))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return gt, out


def _print_table(rows, out):
    """Print an aligned param | gt | est | err | sigma | z | PASS/FAIL table."""
    header = f"{'param':<30}  {'gt':>12}  {'est':>12}  {'err':>12}  {'sigma':>12}  {'z':>8}  status"
    sep = "-" * len(header)
    print(sep, file=sys.stdout)
    print(header, file=sys.stdout)
    print(sep, file=sys.stdout)
    for r in rows:
        status = "PASS" if r["pass_"] else "FAIL"
        z_str = f"{r['z']:8.2f}" if np.isfinite(r["z"]) else "     nan"
        s_str = f"{r['sigma']:12.6f}" if np.isfinite(r["sigma"]) else "         nan"
        print(
            f"{r['param']:<30}  {r['gt']:12.6f}  {r['est']:12.6f}  "
            f"{r['err']:12.6f}  {s_str}  {z_str}  {status}",
            file=sys.stdout,
        )
    print(sep, file=sys.stdout)
    n_fail = out.get("n_fail", sum(1 for r in rows if not r["pass_"]))
    print(f"reproj_rms_px={out.get('reproj_rms_px', float('nan')):.3f}  "
          f"n_fail={n_fail}/{len(rows)}", file=sys.stdout)
