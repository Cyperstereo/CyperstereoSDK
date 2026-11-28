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
"""Canonical IMU noise densities for simulation + the recovery solve.

Single source of truth for the continuous-time IMU noise model so the sim scripts
and the calibration config can't drift. Bridges the naming gap between the two
APIs: `simulate_imu` takes `sigma_w/sigma_bg/sigma_a/sigma_ba` while
`CamImuCalibConfig` takes `sigma_g/sigma_bg/sigma_a/sigma_ba` (gyro density spelled
`sigma_w` on the sim side, `sigma_g` on the solve side).

`MVIS_IMU_NOISE` is a named PRESET, not a hard requirement — CUSTOM noise is
first-class: construct your own `ImuNoiseDensities(...)` and pass `.sim_kwargs()` /
`.calib_kwargs()`, or override the `sigma_*` fields on `SimStudyConfig` /
`CamImuCalibConfig` directly. `scaled(k)` makes noise sweeps (e.g. the S2 study)
one line: `MVIS_IMU_NOISE.scaled(2.0)`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImuNoiseDensities:
    """Continuous-time IMU noise densities.

    sigma_w  : gyro white-noise density       (rad/s/√Hz)
    sigma_bg : gyro bias random-walk          (rad/s²/√Hz)
    sigma_a  : accel white-noise density       (m/s²/√Hz)
    sigma_ba : accel bias random-walk          (m/s³/√Hz)
    """
    sigma_w: float
    sigma_bg: float
    sigma_a: float
    sigma_ba: float

    def sim_kwargs(self) -> dict:
        """Noise kwargs for `simulation.imu_measurements.simulate_imu`."""
        return dict(sigma_w=self.sigma_w, sigma_bg=self.sigma_bg,
                    sigma_a=self.sigma_a, sigma_ba=self.sigma_ba)

    def calib_kwargs(self) -> dict:
        """Noise kwargs for `core.config.CamImuCalibConfig` (gyro density is
        `sigma_g` there, but it is the same physical `sigma_w`)."""
        return dict(sigma_g=self.sigma_w, sigma_bg=self.sigma_bg,
                    sigma_a=self.sigma_a, sigma_ba=self.sigma_ba)

    def scaled(self, k: float) -> "ImuNoiseDensities":
        """All four densities scaled by `k` — for noise sweeps (k=0 → noise-free).
        Pass per-channel factors as a 4-tuple/sequence (w, bg, a, ba) instead."""
        kw = (k, k, k, k) if np.isscalar(k) else tuple(k)
        return ImuNoiseDensities(self.sigma_w * kw[0], self.sigma_bg * kw[1],
                                 self.sigma_a * kw[2], self.sigma_ba * kw[3])


def matched_calib_cfg(noise: "ImuNoiseDensities", *, imu_model: str = "calyx",
                      sigma_px: float = 1.0, huber_px: float = 2.0,
                      imu_intrinsics: str = "dw_da", **over):
    """Build a `CamImuCalibConfig` whose IMU-noise weights MATCH the sim `noise`
    densities (so the estimator is statistically consistent with the data) and whose
    IMU-intrinsic model matches the sim. `**over` sets/overrides any other config
    field (state_hz, sigma_td_prior, init_kwargs, z_min_frac, ...). Shared by
    `SimStudyConfig.calib_cfg` and the multi-board `estimator_config`."""
    from mvis.core.config import CamImuCalibConfig
    kw = dict(sigma_px=sigma_px, huber_px=huber_px, imu_intrinsics=imu_intrinsics,
              imu_intrinsics_model=imu_model, **noise.calib_kwargs())
    kw.update(over)
    return CamImuCalibConfig(**kw)


# MVIS multi-IMU launch densities (yulin_sim_multi_imu.launch / SimulatorCalib);
# previously duplicated in sim_aruco_grounded.py and sim_grounded_scenario.py.
MVIS_IMU_NOISE = ImuNoiseDensities(
    sigma_w=1.6968e-04,
    sigma_bg=1.9393e-05,
    sigma_a=3.0e-03,
    sigma_ba=3.0e-03,
)
