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
"""Shared per-image detection front-end.

ONE place that turns a decoded image into board/feature observations, used by
BOTH the camera-calib detection pipeline and the cam-IMU loader. Wraps the
``Board.detect()`` contract (AprilGrid, ArUco, chessboard, and the geometry-free
``*_features`` detectors all implement it) into the canonical
``FrameObservations`` structure so the two pipelines see *identical* detections.

Also provides:
  * ``frames_to_arrays`` — adapt FrameObservations -> the cam-IMU pipeline's
    ``(t_cam, ids_list, uv_list)`` return shape (keeps that contract unchanged).
  * ``board_signature`` — re-exported from ``io.capture_key`` (its home since F4).
"""
from __future__ import annotations

import numpy as np

from mvis.core.types import FrameObservations
from mvis.io.capture_key import board_signature  # noqa: F401  (old import path)


def detect_frame(image: np.ndarray, board, cam_id: int = 0,
                 timestamp: float = 0.0) -> FrameObservations:
    """Detect one image -> FrameObservations (the shared front-end).

    ``image`` is a decoded uint8 grayscale array. Image size is recorded as
    (width, height). The detector sets per-corner ``point_id``/``pixel``/``row``.
    """
    obs = board.detect(image)
    h, w = image.shape[:2]
    return FrameObservations(camera_id=cam_id, timestamp=float(timestamp),
                             observations=obs, image_size=(int(w), int(h)))


def frames_to_arrays(frames):
    """FrameObservations list -> (t_cam[N], ids_list[N], uv_list[N]).

    The cam-IMU pipeline consumes per-frame ragged ``ids_list`` (int point_ids)
    and ``uv_list`` (Nx2 pixels); this keeps that contract while detection flows
    through the shared FrameObservations structure.
    """
    t_cam = np.asarray([f.timestamp for f in frames], float)
    ids_list = [np.array([o.point_id for o in f.observations], int) for f in frames]
    uv_list = [np.array([o.pixel for o in f.observations], float).reshape(-1, 2)
               for f in frames]
    return t_cam, ids_list, uv_list

