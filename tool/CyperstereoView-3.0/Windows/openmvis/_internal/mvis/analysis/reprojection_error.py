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
"""Reprojection error computation and analysis."""

import numpy as np
import gtsam

from mvis.detection.board import Board
from mvis.camera.camera_model import CameraModel
from mvis.core.types import FrameObservations, ReprojectionErrorStats


def compute_reprojection_errors(
    frames: list[FrameObservations],
    board: Board,
    camera: CameraModel,
    board_poses: list[np.ndarray | None],
    camera_id: int = 0,
) -> ReprojectionErrorStats:
    """Compute reprojection errors for a set of frames.

    Args:
        frames: Frame observations.
        board: Calibration board.
        camera: Camera model with calibrated parameters.
        board_poses: Per-frame 4x4 SE(3) board poses (None for skipped frames).
        camera_id: Camera ID for the result.

    Returns:
        ReprojectionErrorStats.
    """
    errors = []

    for frame, pose_mat in zip(frames, board_poses):
        if pose_mat is None:
            continue

        pose = gtsam.Pose3(pose_mat)

        for obs in frame.observations:
            pt_3d = board.point_3d(obs.point_id)
            # Transform to camera frame
            p_cam = pose.transformFrom(gtsam.Point3(pt_3d[0], pt_3d[1], pt_3d[2]))
            p_cam_np = np.array([p_cam[0], p_cam[1], p_cam[2]])

            # Project through camera model
            projected = camera.project(p_cam_np)
            if np.any(np.isnan(projected)):
                continue

            err = np.linalg.norm(projected - obs.pixel)
            errors.append(err)

    if not errors:
        return ReprojectionErrorStats(
            camera_id=camera_id, rms=0, mean=0, median=0, max=0, num_observations=0
        )

    errors = np.array(errors)
    return ReprojectionErrorStats(
        camera_id=camera_id,
        rms=float(np.sqrt(np.mean(errors ** 2))),
        mean=float(np.mean(errors)),
        median=float(np.median(errors)),
        max=float(np.max(errors)),
        num_observations=len(errors),
    )

