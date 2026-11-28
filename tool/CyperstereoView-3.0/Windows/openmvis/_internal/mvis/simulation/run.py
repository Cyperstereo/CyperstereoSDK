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
"""The one shared sim -> recover -> report pipeline.

`run_sim_recovery(cfg)` resolves GT (by the config's `gt_source`), simulates the
sensors, runs the cam-IMU recovery with a noise-matched estimator, and scores it.
Every study calls this; the only thing that varies is the `SimConfig`.
"""
from __future__ import annotations

from mvis.simulation.noise import MVIS_IMU_NOISE
from mvis.simulation.report import recovery_errors


def run_sim_recovery(cfg, *, pin_intrinsics: bool = False):
    """Simulate + recover + score one `SimConfig`. Returns `(gt, res, errors)`."""
    # lazy import: mvis.calibration imports mvis.simulation, so importing
    # the pipeline at module top would create an import cycle.
    from mvis.calibration.multi_board_pipeline import run_multi_board_calibration

    _scn, cam, imu, board_points, gt, aux_imus = cfg.simulate()
    res = run_multi_board_calibration(
        cam, board_points, imu.t, imu.w_m, imu.a_m,
        config=cfg.estimator_config(), pin_intrinsics=pin_intrinsics,
        aux_imus=aux_imus)
    return gt, res, recovery_errors(gt, res)


def run_digital_twin(gt, board_points, out, *, rates=None, sigma_px=0.5, fps=20,
                     stride=8, names=None):
    """Digital twin: simulate every camera of a `GroundTruth` (`ground_truth_from_result`
    output) and render the multi-camera world video. Returns `(out_path, sims)` where
    `sims` is the per-camera simulated detection cam-dicts (feedable to
    `run_cam_imu_calibration_multi` for a closed-loop check). `rates` is per-camera Hz."""
    import numpy as np
    import gtsam
    from mvis.simulation.multicam import simulate_cameras, traj_from_gt
    from mvis.simulation.video import CalibScene, render_calib_video_multicam

    sims = simulate_cameras(gt, board_points, rates=rates, sigma_px=sigma_px)
    traj = traj_from_gt(gt)
    bp = {0: np.asarray(board_points, float)}
    bpose = {0: gtsam.Pose3(np.asarray(gt.T_BinW, float))}
    cameras = []
    for ci, (cam, s) in enumerate(zip(gt.cams, sims)):
        scene = CalibScene(
            traj=traj, board_points=bp, board_poses=bpose,
            cal_vec=np.asarray(cam['intr8'], float), is_fisheye=bool(cam['is_fisheye']),
            image_size=tuple(cam['image_size']),
            T_CtoI=gtsam.Pose3(np.asarray(cam['T_CtoI'], float)), td=float(cam.get('td', 0.0)),
            readout=float(cam.get('readout', 0.0)), model=s['model'])
        nm = (names[ci] if names else f"cam{ci}") + " (sim)"
        cameras.append(dict(name=nm, scene=scene, det=s))
    path = render_calib_video_multicam(cameras, out, fps=fps, stride=stride, show_trajectory=True,
                                       det_label="simulated (GT)")   # twin corners are simulated
    return path, sims


def sims_to_cams(sims) -> list[dict]:
    """simulate_cameras output -> the camera dicts of run_cam_imu_calibration_multi: intrinsics pinned
    (the GT is the prior), readout estimated only for a rolling-shutter camera."""
    return [dict(t_cam=s['t_cam'], ids_list=s['ids_list'], uv_list=s['uv_list'],
                 cal_vec=s['cal_vec'], is_fisheye=s['is_fisheye'], model=s['model'],
                 image_size=s['image_size'], pin_intrinsics=True,
                 estimate_readout=(s['readout'] > 1e-9), rolling_mode=0) for s in sims]


def _rotdeg(A, B):
    """Geodesic angle (degrees) between two poses' rotations — the SO(3) distance,
    ignoring translation."""
    from mvis.simulation.report import _rot_err_deg
    return _rot_err_deg(A.rotation(), B.rotation())


def gt_vs_recovered(gt, r) -> dict:
    """Errors of a closed-loop recovery `r` vs the GT it was simulated from: per camera the
    T_CtoI rotation (deg) and lever (mm) error, td and readout error (ms) and reprojection RMS;
    per aux IMU the T_IstoI errors; the largest Dw/Da diagonal error of the base IMU."""
    import numpy as np
    import gtsam
    from mvis.imu import ImuModel, parse_imu_model

    out = dict(ok=bool(r.ok), reproj_px=float(r.diagnostics['reproj_rms_px']), cams=[], aux=[])
    for c, e in enumerate(r.cameras):
        cal = e['calib']
        G = gtsam.Pose3(np.asarray(gt.cams[c]['T_CtoI'], float))
        out['cams'].append(dict(
            dR_deg=_rotdeg(G, cal.T),
            dlever_mm=float(1e3 * np.linalg.norm(
                np.asarray(G.translation()) - np.asarray(cal.T.translation()))),
            td_ms=float(cal.td * 1e3),
            readout_err_ms=float(abs(cal.tr - gt.cams[c].get('readout', 0.0)) * 1e3),
            reproj_px=float(e['reproj_rms_px'])))
    GTaux = {m['name']: gtsam.Pose3(np.asarray(m['T_IstoI'], float))
             for m in gt.imus if not m.get('is_base') and m.get('T_IstoI')}
    for m in (r.imus or []):
        if 'T_IstoI' in m and m.get('name') in GTaux:
            G, T = GTaux[m['name']], m['T_IstoI']
            out['aux'].append(dict(name=m['name'], dR_deg=_rotdeg(G, T),
                dlever_mm=float(1e3 * np.linalg.norm(
                    np.asarray(G.translation()) - np.asarray(T.translation()))),
                td_ms=float(1e3 * m.get('td', 0.0))))
    ii = ImuModel.from_tangent(np.asarray(r.imu_intr_xi, float), parse_imu_model(r.imu_model))
    gi = ImuModel.from_tangent(np.asarray(gt.imus[0]['xi24'], float), parse_imu_model(gt.imu_model))
    out['imu_Dw_max'] = float(np.max(np.abs(np.diag(ii.Dw()) - np.diag(gi.Dw()))))
    out['imu_Da_max'] = float(np.max(np.abs(np.diag(ii.Da()) - np.diag(gi.Da()))))
    return out


_gt_vs_recovered = gt_vs_recovered          # old private name, kept for existing callers


def simulate_twin_inputs(gt, board_points, *, rates=None, sigma_px=0.5, seed=0, imu_hz=200.0,
                         imu_noise=MVIS_IMU_NOISE, imu_seed=None, include_aux=True, t_ga=None):
    """The solver inputs of a digital twin of `gt`: `(cams, t_imu, w_m, a_m, aux_imus)`.

    Cameras come from `simulate_cameras` (noise seed `seed`) with pinned intrinsics
    (`sims_to_cams`); the IMU samples are at `imu_hz` from 0.5 s after the first GT state to
    0.5 s before the last. `imu_seed` (default `seed + 100`) seeds the IMU noise:
    - `t_ga=None`: base and aux IMUs from `simulate_imus` (aux k at `imu_seed + 10 + k`);
    - `t_ga` given (s, 0 included): the base accel lags the gyro by t_ga
      (`simulate_base_imu_tga`: gyro at `imu_seed`, accel at `imu_seed + 1`).
    `include_aux=False` returns no aux IMUs. `imu_noise=None` simulates noise-free IMUs."""
    from mvis.simulation.imu_measurements import imu_times
    from mvis.simulation.multicam import simulate_base_imu_tga, simulate_cameras, simulate_imus

    imu_seed = seed + 100 if imu_seed is None else imu_seed
    sims = simulate_cameras(gt, board_points, rates=rates, sigma_px=sigma_px, seed=seed)
    cams = sims_to_cams(sims)
    t_imu = imu_times(float(gt.t_states[0]) + 0.5, float(gt.t_states[-1]) - 0.5, imu_hz)
    if t_ga is None:
        base, aux = simulate_imus(gt, t_imu, noise=imu_noise, seed=imu_seed)
        t, w_m, a_m = base.t, base.w_m, base.a_m
    else:
        t, w_m, a_m = simulate_base_imu_tga(gt, t_imu, noise=imu_noise, seed=imu_seed, t_ga=t_ga)
        aux = simulate_imus(gt, t_imu, noise=imu_noise, seed=imu_seed)[1] if include_aux else []
    return cams, t, w_m, a_m, (aux if include_aux else [])


def digital_twin_closed_loop(gt, board_points, *, solve_config, cfg=None) -> dict:
    """Closed loop: simulate the GT's cameras + IMUs, re-run the multi-camera
    estimator, and return GT-vs-recovered errors (per-camera T_CtoI/td/readout, aux
    T_IstoI, IMU Dw/Da). NOTE: runs a full multi-camera solve (minutes)."""
    import numpy as np
    from mvis.calibration.pipeline_cam_imu import run_cam_imu_calibration_multi
    from mvis.simulation.multicam import DigitalTwinConfig

    cfg = cfg or DigitalTwinConfig()
    cams, t, w_m, a_m, aux = simulate_twin_inputs(
        gt, board_points, rates=cfg.rates, sigma_px=cfg.sigma_px, seed=cfg.seed,
        imu_hz=cfg.imu_hz, imu_noise=cfg.imu_noise)
    r = run_cam_imu_calibration_multi(cams, np.asarray(board_points, float),
                                      t, w_m, a_m, aux_imus=aux, config=solve_config)
    return gt_vs_recovered(gt, r)
