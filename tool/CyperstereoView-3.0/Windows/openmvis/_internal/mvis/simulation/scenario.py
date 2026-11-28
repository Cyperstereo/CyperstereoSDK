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
"""Calibration scenario assembly (simulator Phase 3c / Phase 4 seed).

Bundles a gravity-aligned trajectory + ground-truth calibration + a board placed
in the world into one object that generates a coherent multi-sensor dataset
(camera frames, base IMU, optional auxiliary IMU) with known feature IDs and full
ground truth — the analogue of the C++ MVIS `SimulatorCalib`, and the input the cam-IMU
joint recovery (and eventually `pipeline_cam_imu`) consumes.

Frames: world `W` = the gravity-aligned trajectory frame
(`g_inW = [0,0,-9.81]`, z-up); the board is an object placed in `W` at `T_BinW`
(via `place_board_facing_camera`); camera pose `T_CtoW = T_ItoW(t+td)·T_CtoI`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import gtsam

from mvis.simulation.imu_measurements import simulate_imu, GRAVITY
from mvis.simulation.camera_measurements import (
    simulate_camera, board_points_in_global, place_board_facing_camera,
)


def simulate_base_imu(traj, times, intr, *, g_inW, b_g, b_a, noise=None, seed=0):
    """Base-IMU stream for a scenario (shared by single- and multi-board)."""
    return simulate_imu(traj, times, intr, g_inG=g_inW, b_g0=b_g, b_a0=b_a,
                        seed=seed, **(noise or {}))


def simulate_aux_imu(traj, times, aux, *, g_inW, noise=None, seed=0):
    """Auxiliary-IMU stream (lever arm + intrinsics) — shared helper."""
    return simulate_imu(
        traj, times, aux.intr, g_inG=g_inW,
        R_IstoI=aux.T_IstoI.rotation().matrix(),
        p_IsinI=np.asarray(aux.T_IstoI.translation(), float),
        td=aux.td, b_g0=aux.b_g, b_a0=aux.b_a, seed=seed, **(noise or {}))


class _ImuStreams:
    """Base- and aux-IMU streams of a scenario dataclass with `traj`, `imu_intr`, `b_g`,
    `b_a`, `g_inW`, `aux` and the four noise densities `sigma_w/a/bg/ba`; shared by
    `CalibScenario` and `MultiBoardScenario`."""

    def _noise(self) -> dict:
        """The four IMU noise densities as `simulate_imu` kwargs (all-zero = noise-free)."""
        return dict(sigma_w=self.sigma_w, sigma_a=self.sigma_a,
                    sigma_bg=self.sigma_bg, sigma_ba=self.sigma_ba)

    def base_imu(self, times, seed: int = 0):
        """Base-IMU stream at `times` from the scenario's ground truth (intrinsics,
        biases, gravity) plus `_noise()`. The base sits at the body origin, so it
        sees no lever-arm term — contrast `aux_imu`."""
        return simulate_base_imu(self.traj, times, self.imu_intr, g_inW=self.g_inW,
                                 b_g=self.b_g, b_a=self.b_a, noise=self._noise(), seed=seed)

    def aux_imu(self, times, seed: int = 0):
        """Auxiliary-IMU stream at its lever arm and time offset: the aux feels the extra
        w x (w x p) + alpha x p specific force from the lever, so it carries differential
        info the base IMU lacks. Raises `ValueError` when the scenario has no `aux`."""
        if self.aux is None:
            raise ValueError("scenario has no auxiliary IMU")
        return simulate_aux_imu(self.traj, times, self.aux, g_inW=self.g_inW,
                                noise=self._noise(), seed=seed)


@dataclass
class AuxImu:
    """An auxiliary IMU: extrinsic (aux->base) + time offset + intrinsics + bias."""
    T_IstoI: gtsam.Pose3
    intr: object
    b_g: np.ndarray
    b_a: np.ndarray
    td: float = 0.0


@dataclass
class CalibScenario(_ImuStreams):
    """Ground-truth calibration scenario + measurement generators."""
    traj: object
    board: object
    T_BinW: gtsam.Pose3
    # camera ground truth
    cal: object                       # gtsam Cal3DS2 / Cal3Fisheye
    T_CtoI: gtsam.Pose3
    image_size: tuple
    imu_intr: object                  # base IMU intrinsics
    b_g: np.ndarray
    b_a: np.ndarray
    g_inW: np.ndarray = field(default_factory=lambda: GRAVITY.copy())
    fisheye: bool = False
    td_cam: float = 0.0
    readout: float = 0.0
    rolling_mode: int = 0
    sigma_px: float = 0.0
    sigma_w: float = 0.0      # IMU gyro/accel white-noise densities + bias RW
    sigma_a: float = 0.0
    sigma_bg: float = 0.0
    sigma_ba: float = 0.0
    aux: AuxImu | None = None

    @property
    def points_inW(self) -> np.ndarray:
        """Board corners in the world frame."""
        return board_points_in_global(self.board, self.T_BinW)

    def camera(self, times, seed: int = 0):
        """Project the board into the camera at `times`, honoring `td_cam` and,
        when `readout > 0`, the rolling-shutter row delay. Returns a `CamStream`
        of per-frame corner indices + pixels; only in-image points are kept."""
        return simulate_camera(
            self.traj, times, self.cal, self.T_CtoI, self.points_inW,
            self.image_size, td=self.td_cam, sigma_px=self.sigma_px,
            readout=self.readout, rolling_mode=self.rolling_mode, seed=seed)


def make_scenario(traj, t0: float, depth: float, board, cal, T_CtoI: gtsam.Pose3,
                  imu_intr, b_g, b_a, image_size, *, g_inW=None, fisheye=False,
                  td_cam=0.0, readout=0.0, rolling_mode=0, sigma_px=0.0,
                  sigma_w=0.0, sigma_a=0.0, sigma_bg=0.0, sigma_ba=0.0,
                  aux: AuxImu | None = None) -> CalibScenario:
    """Build a scenario, auto-placing the board facing the camera at `t0`."""
    T_BinW = place_board_facing_camera(traj, t0, depth, board, T_CtoI)
    return CalibScenario(
        traj=traj, board=board, T_BinW=T_BinW, cal=cal, T_CtoI=T_CtoI,
        image_size=image_size, imu_intr=imu_intr,
        b_g=np.asarray(b_g, float), b_a=np.asarray(b_a, float),
        g_inW=GRAVITY.copy() if g_inW is None else np.asarray(g_inW, float),
        fisheye=fisheye, td_cam=td_cam, readout=readout,
        rolling_mode=rolling_mode, sigma_px=sigma_px, sigma_w=sigma_w,
        sigma_a=sigma_a, sigma_bg=sigma_bg, sigma_ba=sigma_ba, aux=aux)
