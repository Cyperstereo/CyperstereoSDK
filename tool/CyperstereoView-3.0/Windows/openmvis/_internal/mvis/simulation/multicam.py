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
"""Multi-camera simulation — the digital-twin sensor model.

`simulate_cameras(gt, board_points)` projects the board through a `GroundTruth`'s
recovered trajectory + per-camera calib (per-camera rolling shutter + noise), giving
one detection cam-dict per camera. Pair with `render_calib_video_multicam` (video) or
feed back into `run_cam_imu_calibration_multi` for a closed-loop GT-vs-recovered check.
The board geometry is a separate input (it's the board MODEL, not recovered state).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import gtsam

from mvis.simulation.camera_measurements import simulate_camera
from mvis.simulation.imu_measurements import imu_times, simulate_imu
from mvis.simulation.from_result import (
    cal_from_vec, fit_bspline_from_states, gt_cam_model)
from mvis.simulation.noise import MVIS_IMU_NOISE


@dataclass
class DigitalTwinConfig:
    """Knobs for the result-driven digital twin (the 'controlled by config' surface)."""
    rates: list | None = None              # per-camera frame rate (Hz); None = 30 all
    sigma_px: float = 0.5                   # corner noise
    imu_hz: float = 200.0
    imu_noise: object = field(default_factory=lambda: MVIS_IMU_NOISE)
    seed: int = 0
    fps: int = 20                           # video
    stride: int = 8
    recover: bool = False                   # also run the closed-loop GT-vs-recovered check


def traj_from_gt(gt):
    """B-spline through a GroundTruth's IMU-world state poses (T_ItoW)."""
    spline, _err = fit_bspline_from_states(
        np.asarray(gt.t_states, float),
        [np.asarray(T, float) for T in gt.T_states], 0.05)
    return spline


def simulate_cameras(gt, board_points, *, rates=None, sigma_px=0.5, seed=0, margin=0.5):
    """Simulate per-camera detections from a `GroundTruth` + board model. Returns a
    list parallel to `gt.cams`, each a cam-dict: `t_cam`, `ids_list`, `uv_list`,
    `cal_vec`, `is_fisheye`, `model`, `image_size`, `td`, `readout`. `rates` is a
    per-camera frame rate (Hz, default 30); rolling shutter uses each camera's `readout`."""
    traj = traj_from_gt(gt)
    T_BinW = gtsam.Pose3(np.asarray(gt.T_BinW, float))
    pts_W = np.array([np.asarray(T_BinW.transformFrom(gtsam.Point3(*p)), float)
                      for p in np.asarray(board_points, float)])
    t0 = float(gt.t_states[0]) + margin
    t1 = float(gt.t_states[-1]) - margin
    out = []
    for c, cam in enumerate(gt.cams):
        model = gt_cam_model(cam)
        is_fish = model == "equidistant"
        cal = cal_from_vec(cam['intr8'], is_fish, model)   # gtsam Cal3 or CameraModel
        T_CtoI = gtsam.Pose3(np.asarray(cam['T_CtoI'], float))
        rate = float(rates[c]) if rates is not None else 30.0
        tcam = imu_times(t0, t1, rate)
        st = simulate_camera(traj, tcam, cal, T_CtoI, pts_W, tuple(cam['image_size']),
                             td=float(cam.get('td', 0.0)), sigma_px=sigma_px,
                             readout=float(cam.get('readout', 0.0)), rolling_mode=0, seed=seed + c)
        out.append(dict(
            t_cam=np.asarray(st.t, float),
            ids_list=[np.asarray(i, int) for i in st.ids],
            uv_list=[np.asarray(u, float) for u in st.uv],
            cal_vec=np.asarray(cam['intr8'], float), is_fisheye=is_fish,
            model=model,
            image_size=tuple(cam['image_size']), td=float(cam.get('td', 0.0)),
            readout=float(cam.get('readout', 0.0))))
    return out


def simulate_imus(gt, t_imu, *, noise=MVIS_IMU_NOISE, seed=0):
    """Simulate base + aux IMU streams from a `GroundTruth`. Returns `(base, aux_list)`:
    `base` is an ImuStream; `aux_list` is `[{t_imu, w_m, a_m, name}]` for the aux IMUs
    (each at its `T_IstoI` lever)."""
    from mvis.imu import ImuModel, parse_imu_model

    traj = traj_from_gt(gt)
    model = parse_imu_model(gt.imu_model)
    g = np.asarray(gt.g_inW, float)
    nk = noise.sim_kwargs() if noise is not None else {}
    base, aux = None, []
    for i, m in enumerate(gt.imus):
        intr = ImuModel.from_tangent(np.asarray(m['xi24'], float), model)
        if m.get('is_base'):
            base = simulate_imu(traj, t_imu, intr, g_inG=g, seed=seed, **nk)
        else:
            T = gtsam.Pose3(np.asarray(m['T_IstoI'], float))
            st = simulate_imu(traj, t_imu, intr, g_inG=g,
                              R_IstoI=T.rotation().matrix(),
                              p_IsinI=np.asarray(T.translation(), float),
                              td=float(m.get('td', 0.0)), seed=seed + 10 + i, **nk)
            aux.append(dict(t_imu=st.t, w_m=st.w_m, a_m=st.a_m, name=m.get('name', f'aux{i}')))
    return base, aux


def simulate_base_imu_tga(gt, t_imu, *, noise=MVIS_IMU_NOISE, seed=0, t_ga=0.0):
    """Base-IMU stream `(t, w_m, a_m)` with the accel lagging the gyro by `t_ga` (s): the gyro is
    the trajectory at t (noise seed `seed`), the accel the trajectory at t - t_ga from a second
    stream (noise seed `seed + 1`), both stamped t. Positive t_ga = accel lags gyro, the ACI sign."""
    from mvis.imu import ImuModel, parse_imu_model

    traj = traj_from_gt(gt)
    intr = ImuModel.from_tangent(np.asarray(gt.imus[0]['xi24'], float),
                                 parse_imu_model(gt.imu_model))
    g = np.asarray(gt.g_inW, float)
    nk = noise.sim_kwargs() if noise is not None else {}
    A = simulate_imu(traj, t_imu, intr, g_inG=g, seed=seed, **nk)          # gyro
    B = simulate_imu(traj, t_imu, intr, g_inG=g, td=-t_ga, seed=seed + 1, **nk)
    return A.t, A.w_m, B.a_m
