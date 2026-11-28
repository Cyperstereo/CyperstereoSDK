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
"""Driver: multi-board cam-IMU calibration on a single base camera.

Mirrors the single-board `run_cam_imu_calibration_multi` but (1) seeds from the
hybrid `init_multi_board`, (2) carries one board pose per board, and (3) splits
each frame's corners BY BOARD (a wide fisheye frame sees several boards) so each
group reprojects against its own board pose.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import gtsam

from mvis.calibration.cam_imu_init import initialize_imu_imu
from mvis.calibration.factors.camera_calib import CameraCalib
from mvis.calibration.pipeline_cam_imu import (
    build_joint_graph, board_poses_from_values, _decimate, _assign_brackets,
    _densify_velocities, _cam_factor_hz, _cam_model, _cam_pin_tr,
    CamImuCalibConfig, CamImuCalibResult, I0, _imu_model, _lm_params, _reproj_corner_norms, _bias_fields)
from mvis.calibration.multi_board_init import init_multi_board
from mvis.detection.corner_subsample import corner_mask
from mvis.simulation.multi_board import board_of, local_of


def run_multi_board_calibration(cam, board_points_by_id, t_imu, w_m, a_m, *,
                                config=None, anchor_board=0, pin_intrinsics=True,
                                aux_imus=None):
    """Run multi-board cam-IMU calibration: initialize from the detections (init_multi_board), then jointly optimize camera intrinsics/extrinsics/td + IMU intrinsics over ALL boards (anchored at `anchor_board`). `aux_imus` adds IMU-IMU factors. Returns a CamImuCalibResult."""
    cfg = config or CamImuCalibConfig()
    cal_vec = np.asarray(cam['cal_vec'], float)
    model = _cam_model(cam)

    seed, board_poses_init, init_report = init_multi_board(
        cam['t_cam'], cam['ids_list'], cam['uv_list'], board_points_by_id,
        cal_vec, t_imu, w_m, a_m, anchor_board=anchor_board, model=model,
        cfg=cfg, **cfg.init_kwargs)

    # optional extra IMUs: I6 init each against the base gyro (lever arm +
    # intrinsics estimated in the joint graph via the Multi_In_V factor)
    aux_data = []
    for aux in (aux_imus or []):
        sd = initialize_imu_imu(t_imu, w_m, aux['t_imu'], aux['w_m'], b_g_base=seed.b_g)
        aux_data.append({**aux, 'seed': sd})

    # states: the hybrid init's WHOLE-BAG camera-pose track (covers all boards,
    # not just board-0 windows), decimated to state_hz.
    t_track, T_track = init_report['state_track']
    keep_s = _decimate(np.asarray(t_track, float), cfg.state_hz)
    t_states = np.asarray(t_track, float)[keep_s]
    T_init = [T for T, k in zip(T_track, keep_s) if k]
    g_inW = np.array([0.0, 0.0, -cfg.gravity])
    v_init = _densify_velocities(t_states, T_init, seed, t_imu, w_m, a_m, cfg, g_inW)

    # frames -> bracketing states; a frame may see MULTIPLE boards -> one
    # factor-group per (frame, board), each against its own board pose.
    t_cam = np.asarray(cam['t_cam'], float)
    keep = _decimate(t_cam, _cam_factor_hz(cam, cfg))
    t_frames = t_cam[keep]
    frame_idx = np.nonzero(keep)[0]
    j, dt_frame, ok_f = _assign_brackets(t_frames, seed.td, t_states, cfg.max_interp_dt)
    frames, board_ids = [], []
    used_ts = set()
    for m, i in enumerate(frame_idx):
        if not ok_f[m]:
            continue
        ids = np.asarray(cam['ids_list'][i], int)
        uv = np.asarray(cam['uv_list'][i], float).reshape(-1, 2)
        if not len(ids):
            continue
        for b in np.unique(board_of(ids)):
            sel = board_of(ids) == b
            ids_b, uv_b = ids[sel], uv[sel]
            cm = corner_mask(ids_b, cfg.corner_subsample)
            if np.sum(cm) < 6:
                continue
            pts_b = np.asarray(board_points_by_id[int(b)], float)[local_of(ids_b)[cm]]
            frames.append((uv_b[cm], pts_b, int(j[m]), float(dt_frame[m])))
            board_ids.append(int(b))
            used_ts.add(float(t_frames[m]))

    # By default camera intrinsics are GIVEN (already calibrated) -> PINNED; we
    # estimate only the cam-IMU spatial-temporal, IMU intrinsics/biases, and the
    # board poses. Set pin_intrinsics=False to ALSO estimate camera intrinsics.
    # (tr pinned: global-shutter sim.)
    calib_bases = [CameraCalib(seed.T_CtoI, seed.td, 0.0, cal_vec, model=model)]
    # This pipeline reprojects through the two-state chord (_reproj below), so it keeps the interp
    # factor whatever solve.vision_pose says.
    cfg = dataclasses.replace(cfg, vision_pose="interp")
    graph, values, cam_factors, imu_factors, _aux = build_joint_graph(
        t_states, [frames], T_init, v_init, seed, calib_bases,
        [cam['image_size']], t_imu, w_m, a_m, cfg, structure="board",
        board_ids_per_cam=[board_ids], board_poses_init=board_poses_init,
        pin_intr=[pin_intrinsics], pin_trs=[_cam_pin_tr(cam, cfg)],
        rolling_modes=[int(cam.get('rolling_mode', 0))], aux_data=aux_data)

    # multi-board reprojection RMS: each factor reprojects against ITS board pose (its own
    # make_factor keys). Kalibr convention: RMS over the pooled per-corner norms of all frames.
    from .reproj_stats import corner_stats

    def _reproj(vals):
        s = corner_stats(_reproj_corner_norms(cam_factors, vals)[0])
        return s['rms'] if s['n_corners'] else np.inf

    rms_pre = _reproj(values)
    res = gtsam.LevenbergMarquardtOptimizer(graph, values, _lm_params(cfg)).optimize()
    rms_post = _reproj(res)
    board_ids_all = sorted(board_poses_init.keys())
    bp = board_poses_from_values(res, board_ids_all)
    base = calib_bases[0].retract(res.atVector(gtsam.symbol('c', 0)))
    # estimated IMU intrinsics (24-dim xi around identity) + bias drift, read out
    # of the solve (these ARE estimated when cfg.imu_intrinsics != "pinned").
    imu_intr_xi = np.asarray(res.atVector(I0), float)
    bg0, ba0 = seed.b_g, seed.b_a
    diagnostics = {
        'gravity': cfg.gravity, 'gravity_source': cfg.gravity_source, 'gravity_site': cfg.gravity_site,
        'gravity_place': cfg.gravity_place,
        'reproj_rms_px': float(rms_post),
        'reproj_rms_px_init': float(rms_pre),
        'n_states': int(len(t_states)), 'n_groups': int(len(board_ids)),
        'imu_model': _imu_model(cfg), 'imu_intrinsics_mode': str(cfg.imu_intrinsics),
        'board_poses': {b: bp[b].matrix().tolist() for b in board_ids_all},
        'board_poses_init': {b: T.matrix().tolist()
                             for b, T in board_poses_init.items()},
        'init_report': init_report,
    }
    # the count, in diagnostics because this result has no `cameras` list; kb4: log the
    # fit's valid domain (WARNING when it folds inside the image)
    from mvis.calibration.pipeline_cam_imu import (
        _corner_accounting, _frame_fate, _guarded_accounting, _note_kb_domain)
    accounting = _guarded_accounting(1, lambda: _corner_accounting(
        cam_factors, res, [base], [cam['image_size']], [_frame_fate(cam, t_frames, used_ts)]))[0]
    diagnostics['corner_accounting'] = accounting
    _note_kb_domain({'calib': base, 'image_size': cam['image_size'],
                     'corner_accounting': accounting}, 0)
    return CamImuCalibResult(
        calib=base, calib_sigma=np.zeros(8 + base.n_intr),
        **_bias_fields(res, len(t_states)),
        imu_intr_xi=imu_intr_xi, imu_intr_sigma=None,
        T_BinW=bp[anchor_board], seed=seed, values=res,
        t_states=t_states, diagnostics=diagnostics,
        # gate was 2 px in the old per-axis convention; corner norms read
        # sqrt(2) higher for the same residuals, so scale to keep it equivalent
        ok=bool(rms_post < 2.0 * np.sqrt(2)))
