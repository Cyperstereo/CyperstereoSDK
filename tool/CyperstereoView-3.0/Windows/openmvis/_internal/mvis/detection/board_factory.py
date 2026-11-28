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
"""Factory for creating calibration boards from configuration."""

from mvis.detection.board import Board
from mvis.core.config import BoardConfig


def create_board(config: BoardConfig) -> Board:
    """Create a Board instance from configuration.

    Args:
        config: Board configuration.

    Returns:
        Board instance.
    """
    if config.board_type == "chessboard":
        from mvis.detection.chessboard import Chessboard

        return Chessboard(
            rows=config.rows,
            cols=config.cols,
            square_size=config.square_size,
        )
    elif config.board_type == "apriltag":
        from mvis.detection.apriltag_board import AprilTagBoard

        # tag_spacing: if < 1.0, treat as ratio (Kalibr convention)
        spacing = config.tag_spacing
        if spacing < 1.0:
            spacing = config.tag_size * spacing

        return AprilTagBoard(
            rows=config.rows,
            cols=config.cols,
            tag_size=config.tag_size,
            tag_spacing=spacing,
            tag_family=config.tag_family,
        )
    elif config.board_type == "aruco":
        from mvis.detection.aruco_board import ArucoBoard

        return ArucoBoard(
            rows=config.rows,
            cols=config.cols,
            marker_size=config.marker_size,
            marker_separation=config.marker_separation,
            aruco_dict=config.aruco_dict,
        )
    elif config.board_type == "charuco":
        from mvis.detection.charuco_board import CharucoBoard

        # rows/cols count SQUARES here (as board generators print them), not inner corners
        return CharucoBoard(
            rows=config.rows,
            cols=config.cols,
            square_size=config.square_size,
            marker_size=config.marker_size,
            aruco_dict=config.aruco_dict,
            legacy_pattern=config.charuco_legacy,
            refine=config.charuco_refine,
        )
    elif config.board_type == "aruco_features":
        # ArUco tags as geometry-free features (tags on several boards at
        # unknown poses) — for feature-mode (target-free) calibration only.
        from mvis.detection.aruco_features import ArucoFeatureDetector

        return ArucoFeatureDetector(
            aruco_dict=config.aruco_dict,
            min_tag_px=config.detection.min_tag_px,
            refine_subpixel=config.refine_subpixel,
        )
    elif config.board_type == "apriltag_features":
        # AprilTags as geometry-free features — feature-mode calibration only.
        from mvis.detection.apriltag_features import AprilTagFeatureDetector

        return AprilTagFeatureDetector(
            tag_family=config.tag_family,
            min_tag_px=config.detection.min_tag_px,
            refine_subpixel=config.refine_subpixel,
        )
    else:
        raise ValueError(f"Unknown board type: {config.board_type}")
