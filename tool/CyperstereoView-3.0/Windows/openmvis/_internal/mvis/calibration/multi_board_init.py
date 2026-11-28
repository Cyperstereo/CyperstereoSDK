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
"""Multi-board cam-IMU initialization.

NOT purely PnP: PnP gives the board poses + camera-pose track; the cam-IMU state
comes from an inertial alignment (PnP cannot see the IMU). Stages:
  A. Anchor seed on board 0 — `initialize_cam_imu` on board 0's PnP pose track +
     raw IMU recovers T_CtoI, td, gravity, biases, per-keyframe IMU states/vels.
  B. Flip-robust per-board PnP tracks (multi-frame `pnp_pose_track`).
  C. Localize the other boards by INCREMENTAL co-visibility chaining: a board
     co-visible with ANY already-localized board is localized directly (robust
     average). Boards with no co-visibility chain fall back to the IMU bridge
     (propagate from the anchor) — without the IMU their relative pose is
     unobservable, so the bridge is what makes the non-co-visible case possible.
  D. Whole-bag IMU-state track from the localized boards' PnP (+ IMU velocities).
The joint LM solve then refines everything together.
"""
from __future__ import annotations

import numpy as np
import gtsam

from mvis.core.config import CamImuCalibConfig
from mvis.calibration.cam_imu_init import pnp_pose_track, initialize_cam_imu
from mvis.calibration.pipeline_cam_imu import _propagate_state
from mvis.simulation.multi_board import board_of, local_of


def _split_by_board(ids_list, uv_list):
    """Return {board_id: (frame_idx_list, local_ids_list, uv_list)}."""
    out = {}
    for i, (ids, uv) in enumerate(zip(ids_list, uv_list)):
        ids = np.asarray(ids, int)
        if not len(ids):
            continue
        for b in np.unique(board_of(ids)):
            m = board_of(ids) == b
            d = out.setdefault(int(b), ([], [], []))
            d[0].append(i)
            d[1].append(local_of(ids[m]))
            d[2].append(np.asarray(uv)[m])
    return out


def _chordal_average(poses, trans_tol=0.10):
    """Robust pose average. Per-frame single-frame PnP can flip on a planar board,
    and IMU bridging can diverge on a bad window — both produce extreme outliers
    that a plain mean cannot survive. Center on the per-axis MEDIAN translation
    (breakdown 50%), keep only poses within `trans_tol` of it, then chordal-average
    the survivors (translation mean + rotation tangent mean about the survivor
    closest to the median)."""
    T = np.array([np.asarray(p.translation(), float) for p in poses])
    med = np.median(T, axis=0)
    keep = np.nonzero(np.linalg.norm(T - med, axis=1) <= trans_tol)[0]
    if len(keep) == 0:                       # all far from median -> keep nearest
        keep = [int(np.argmin(np.linalg.norm(T - med, axis=1)))]
    sub = [poses[i] for i in keep]
    t = np.mean([np.asarray(p.translation(), float) for p in sub], axis=0)
    R0 = sub[int(np.argmin(np.linalg.norm(T[keep] - med, axis=1)))].rotation()
    xi = np.mean([gtsam.Rot3.Logmap(R0.inverse().compose(p.rotation()))
                  for p in sub], axis=0)
    R = R0.compose(gtsam.Rot3.Expmap(xi))
    return gtsam.Pose3(R, gtsam.Point3(*t))


def init_multi_board(t_cam, ids_list, uv_list, board_points_by_id, cal_vec,
                     t_imu, w_m, a_m, *, anchor_board=0, model="radtan",
                     cfg=None, **init_kwargs):
    """Initialize a multi-board cam-IMU problem: per-board SfM/PnP seeds, inter-board anchoring to `anchor_board` (which defines the world frame), and the camera-IMU spatial-temporal init (rotation, lever, td). Returns (seed, board_poses_init, report)."""
    cfg = cfg or CamImuCalibConfig()
    t_cam = np.asarray(t_cam, float)
    by_board = _split_by_board(ids_list, uv_list)
    if anchor_board not in by_board:
        raise ValueError(f"anchor board {anchor_board} never observed")

    # --- A. anchor seed on board `anchor_board` ---
    a_fi, a_loc, a_uv = by_board[anchor_board]
    pts0 = np.asarray(board_points_by_id[anchor_board], float)
    t_kept, T_CtoB0_kept, _n = pnp_pose_track(t_cam[a_fi], a_loc, a_uv, pts0,
                                              cal_vec, model=model)
    seed = initialize_cam_imu(t_kept, T_CtoB0_kept, t_imu, w_m, a_m, g_mag=cfg.gravity, **init_kwargs)

    T_CtoI = seed.T_CtoI
    T_CtoI_inv = T_CtoI.inverse()
    t_key = np.asarray(seed.t_key, float)
    T_ItoW_key = list(seed.T_ItoW)
    v_key = np.asarray(seed.v_inW, float)
    g_inW = np.array([0.0, 0.0, -cfg.gravity])

    # --- B. FLIP-ROBUST per-board PnP tracks: pnp_pose_track resolves the planar
    # two-fold flip via multi-frame consistency. pose_of[b][frame] = T_CtoBb. ---
    pose_of = {}
    for b, (fi_b, loc_b, uv_b) in by_board.items():
        ptsb = np.asarray(board_points_by_id[b], float)
        fi_arr = np.asarray(fi_b, int)
        try:
            tk_b, Tk_b, _n = pnp_pose_track(t_cam[fi_arr], loc_b, uv_b, ptsb,
                                            cal_vec, model=model)
        except ValueError:
            continue
        ta = t_cam[fi_arr]
        d = {}
        for tt, T in zip(tk_b, Tk_b):
            d[int(fi_arr[int(np.argmin(np.abs(ta - float(tt))))])] = T
        pose_of[int(b)] = d
    pose_of[anchor_board] = {int(fi): T for fi, T in
                             pose_of.get(anchor_board, {}).items()}

    # --- C. INCREMENTAL co-visibility localization (fisheye-robust, bridge-free
    # when a chain exists): anchor is known; localize any board co-visible with an
    # already-localized board via T_BbinW = (T_LinW . T_CtoL) . T_CtoBb^-1. ---
    board_poses = {int(anchor_board): seed.T_BinW}
    report = {'anchor': int(anchor_board), 'per_board': {}}
    changed = True
    while changed:
        changed = False
        for b in list(pose_of):
            if b in board_poses:
                continue
            ests = []
            for i, T_CtoBb in pose_of[b].items():
                for L, TL in board_poses.items():
                    if i in pose_of.get(L, {}):
                        T_cw = TL.compose(pose_of[L][i])     # camera in W via board L
                        ests.append(T_cw.compose(T_CtoBb.inverse()))  # T_BbinW
                        break
            if ests:
                board_poses[int(b)] = _chordal_average(ests)
                report['per_board'][int(b)] = {'n_frames': len(ests), 'mode': 'direct'}
                changed = True

    # any board with no co-visibility chain -> IMU bridge fallback
    def _bridge(t):
        k = int(np.clip(np.searchsorted(t_key, t) - 1, 0, len(t_key) - 1))
        T_ItoW, _v = _propagate_state(T_ItoW_key[k], v_key[k], t_key[k],
                                      float(t) + seed.td, t_imu, w_m, a_m,
                                      seed.b_g, seed.b_a, cfg, g_inW)
        return T_ItoW.compose(T_CtoI)

    for b in list(pose_of):
        if b in board_poses:
            continue
        ests = [_bridge(t_cam[i]).compose(T.inverse()) for i, T in pose_of[b].items()]
        if ests:
            board_poses[int(b)] = _chordal_average(ests)
            report['per_board'][int(b)] = {'n_frames': len(ests), 'mode': 'bridge'}

    # --- D. whole-bag IMU-state track from the localized boards' robust PnP. For
    # each frame use a localized board present in it (prefer the anchor); bridge
    # only frames with no localized board. T_ItoW = T_CtoW . T_CtoI^-1. ---
    all_frames = sorted({int(i) for d in pose_of.values() for i in d})
    t_states, T_states = [], []
    for i in all_frames:
        Ls = [L for L in board_poses if i in pose_of.get(L, {})]
        if Ls:
            L = anchor_board if anchor_board in Ls else Ls[0]
            T_cw = board_poses[L].compose(pose_of[L][i])
        else:
            T_cw = _bridge(t_cam[i])
        t_states.append(t_cam[i])
        T_states.append(T_cw.compose(T_CtoI_inv))
    report['state_track'] = (np.array(t_states, float), T_states)
    return seed, board_poses, report
