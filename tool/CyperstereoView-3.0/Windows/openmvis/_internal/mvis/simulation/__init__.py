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
"""mvis.simulation — synthetic camera-IMU data with known ground truth.

A Python reimplementation, written from the published math, of the calibration
simulator used in MVIS (C++) and OpenVINS: an SE(3) B-spline trajectory drives
analytic IMU measurements and feature projections, so the calibration pipeline
can be scored against ground truth.
"""

from mvis.simulation.bspline_se3 import BsplineSE3
from mvis.simulation.imu_measurements import (
    ImuStream, simulate_imu, measure_imu, true_imu, aux_imu_true, imu_times,
    GRAVITY,
)
from mvis.simulation.trajectory_io import load_trajectory
from mvis.simulation.camera_measurements import (
    CamStream, simulate_camera, project_points, project_points_rolling,
    board_points_in_global, place_board_facing_camera,
)
from mvis.simulation.scenario import (
    CalibScenario, AuxImu, make_scenario, simulate_base_imu, simulate_aux_imu,
)
from mvis.simulation.noise import (
    MVIS_IMU_NOISE, ImuNoiseDensities, matched_calib_cfg,
)
from mvis.simulation.multi_board import SimConfig, MultiBoardSimConfig
from mvis.simulation.report import recovery_errors, format_recovery_table
from mvis.simulation.run import run_sim_recovery, run_digital_twin
from mvis.simulation.from_result import ground_truth_from_result
from mvis.simulation.multicam import simulate_cameras
from mvis.simulation.visualization import plot_scenario
from mvis.simulation.video import (
    render_sim_video, render_calib_video, render_calib_video_from_path,
    CalibScene, calib_scene_from_result, board_points_from, render_detections_video,
)

__all__ = [
    "BsplineSE3",
    "ImuStream", "simulate_imu", "measure_imu", "true_imu", "aux_imu_true",
    "imu_times", "GRAVITY",
    "load_trajectory",
    "CamStream", "simulate_camera", "project_points", "project_points_rolling",
    "board_points_in_global", "place_board_facing_camera",
    "CalibScenario", "AuxImu", "make_scenario", "simulate_base_imu", "simulate_aux_imu",
    "MVIS_IMU_NOISE", "ImuNoiseDensities", "matched_calib_cfg",
    "SimConfig", "MultiBoardSimConfig",
    "recovery_errors", "format_recovery_table", "run_sim_recovery",
    "run_digital_twin", "ground_truth_from_result", "simulate_cameras",
    "plot_scenario",
    "render_sim_video", "render_calib_video", "render_calib_video_from_path",
    "CalibScene", "calib_scene_from_result", "board_points_from", "render_detections_video",
]
