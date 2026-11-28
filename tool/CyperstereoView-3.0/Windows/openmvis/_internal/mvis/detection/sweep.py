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
"""sweep — stage ① of the run: every frame, every enabled
detector, filling the frozen measurement core of an MVISDatabase.

Contract:
  * frames are consumed IN TIME ORDER PER CAMERA (a future stateful
    FreeFeatureDetector depends on this — never parallelize per-frame);
  * one image in memory at a time (run_streaming's discipline);
  * the sweep writes ONLY measurements: frames, detection groups, detections.
    No caching (the manager's job), no subsampling (run policy, stage ②),
    no thumbnails (analysis').
"""
from __future__ import annotations

from mvis.detection.detectors import BoardFeatureDetector, FeatureDetector
from mvis.mvisdb.mvisdatabase import DetectionMode, MVISDatabase


def sweep(source, detectors: list[FeatureDetector], db: MVISDatabase, *,
          min_obs: int = 0) -> int:
    """Iterate ``source`` — an iterable of (image, timestamp, camera_id) — and
    fill ``db``. Returns the number of frames stored.

    ``min_obs`` mirrors today's loader behavior of skipping frames with too few
    observations at INGEST time; pass 0 (the design default) to keep every
    frame and let run policy select instead.
    """
    n = 0
    for image, timestamp, cam_id in source:
        per_detector = [(d, d.detect(image, timestamp)) for d in detectors]
        total = sum(len(dets) for _, dets in per_detector)
        if total < min_obs:
            continue
        frame = db.add_frame(cam_id, float(timestamp))
        n += 1
        for d, dets in per_detector:
            if d.mode is DetectionMode.BOARD and isinstance(d, BoardFeatureDetector):
                db.add_board_detection(frame, d.board_id, dets)
            elif dets:
                # tag/free detections have no schema home yet — loud until they
                # exist, never silently dropped
                raise NotImplementedError(
                    f"{d.mode} detections have no schema home yet")
    return n
