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
"""detection.boards — the component's target facade.

The geometry-full targets (AprilTag grid, ArUco grid, chessboard, ChArUco) and
`create_board`, re-exported from their implementation modules so that user code
can ``from mvis.detection.boards import create_board`` without knowing the
layout. Type new code against BoardGeometry (the target contract), not the
legacy Board.
"""
from mvis.detection.board import Board, BoardGeometry            # noqa: F401
from mvis.detection.apriltag_board import AprilTagBoard          # noqa: F401
from mvis.detection.aruco_board import ArucoBoard                # noqa: F401
from mvis.detection.chessboard import Chessboard                 # noqa: F401
from mvis.detection.charuco_board import CharucoBoard            # noqa: F401
from mvis.detection.board_factory import create_board            # noqa: F401

__all__ = ["Board", "BoardGeometry", "AprilTagBoard", "ArucoBoard",
           "Chessboard", "CharucoBoard", "create_board"]
