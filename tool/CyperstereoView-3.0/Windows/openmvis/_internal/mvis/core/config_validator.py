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
"""Validate a CalibrationConfig for common errors and inconsistencies.

`validate_config` is the library-level check of a parsed `CalibrationConfig`
(board, cameras, optimizer ranges); it returns the problems as strings instead of
raising. The `mvis check` subcommand runs its own file-level checks
(`mvis.doctor.check_config`); call this one from code that builds a config
programmatically, e.g. ``validate_config(CalibrationConfig.from_yaml(path))``.
"""

from __future__ import annotations

from mvis.core.config import CalibrationConfig
from mvis.core.camera_models import MODELS


def validate_config(config: CalibrationConfig) -> list[str]:
    """Return a list of error/warning strings. Empty = valid."""
    errors: list[str] = []

    # --- Board ---
    b = config.board
    if b.rows < 2 or b.cols < 2:
        errors.append("Board must have at least 2 rows and 2 cols")
    if b.tag_size <= 0 and b.board_type == "apriltag":
        errors.append("tag_size must be positive")
    if b.square_size <= 0 and b.board_type in ("chessboard", "charuco"):
        errors.append("square_size must be positive")
    if b.board_type == "charuco":
        if b.rows < 3 or b.cols < 3:
            errors.append("ChArUco rows and cols count squares and must be at least 3")
        if not 0 < b.marker_size < b.square_size:
            errors.append("ChArUco marker_size must be positive and smaller than square_size")
        from mvis.detection.charuco_board import dictionary_problem
        problem = dictionary_problem(b.aruco_dict, b.rows, b.cols)
        if problem:
            errors.append(problem)
    # corner_subsample is a top-level camera-only field.
    if config.corner_subsample not in (
        "all", "top_left", "tl", "bl", "br", "tr",
        "bottom_left", "bottom_right", "top_right",
        "two_diag", "two_anti",
    ):
        errors.append(f"Invalid corner_subsample: '{config.corner_subsample}'")
    if b.detection.early_stop_rate < 0.5:
        errors.append("detection.early_stop_rate < 0.5 is too aggressive")

    # --- Cameras ---
    if not config.cameras:
        errors.append("No cameras defined")
    cam_ids = [c.camera_id for c in config.cameras]
    if len(cam_ids) != len(set(cam_ids)):
        errors.append("Duplicate camera_id values")
    if config.cameras and config.reference_camera not in cam_ids:
        errors.append(
            f"reference_camera {config.reference_camera} "
            f"not in camera list {cam_ids}"
        )
    for cam in config.cameras:
        if cam.factor_hz <= 0:
            errors.append(
                f"cam{cam.camera_id}: factor_hz must be positive "
                f"(got {cam.factor_hz})"
            )
        if cam.shutter_type not in ("auto", "rolling", "global"):
            errors.append(
                f"cam{cam.camera_id}: invalid shutter_type "
                f"'{cam.shutter_type}'"
            )
        if cam.distortion_model not in MODELS:
            errors.append(
                f"cam{cam.camera_id}: invalid distortion_model "
                f"'{cam.distortion_model}'"
            )

    # --- Optimizer ---
    o = config.optimizer
    if o.pixel_sigma <= 0:
        errors.append("optimizer.pixel_sigma must be positive")
    if o.huber_threshold <= 0:
        errors.append("optimizer.huber_threshold must be positive")
    if o.anchor_prior_sigma <= 0:
        errors.append("optimizer.anchor_prior_sigma must be positive")
    if o.guardrail.p95_regression_factor <= 1.0:
        errors.append(
            "guardrail.p95_regression_factor must be > 1.0 "
            f"(got {o.guardrail.p95_regression_factor})"
        )
    if o.guardrail.fx_min < 0:
        errors.append("guardrail.fx_min must be non-negative")
    if o.outlier_rejection.threshold_px <= 0:
        errors.append("outlier_rejection.threshold_px must be positive")

    # --- RS warning ---
    ref_cam = next(
        (c for c in config.cameras if c.camera_id == config.reference_camera),
        None,
    )
    if ref_cam and ref_cam.shutter_type == "rolling":
        errors.append(
            "WARNING: reference camera is rolling shutter — consider "
            "using a global-shutter camera as reference for better "
            "RS calibration on non-ref cameras"
        )

    # Single-cam RS warning
    rs_cams = [c for c in config.cameras if c.shutter_type == "rolling"]
    if rs_cams and len(config.cameras) == 1:
        errors.append(
            "WARNING: single-camera RS calibration is degenerate — "
            "RS readout cannot be estimated without a GS reference camera. "
            "Set rolling_shutter_readout as a known constant instead."
        )

    return errors
