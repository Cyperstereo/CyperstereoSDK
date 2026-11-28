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
"""Bridge: today's pipeline shapes <-> MVISDatabase (transitional by design).

Existing pipelines keep their `FrameObservations` lists and `(t_cam, ids_list, uv_list)`
ragged tuples; this module converts either into/out of the structure so the
fingerprint harness can prove bit-identity BEFORE any pipeline is rewired.
When the pipelines take `MVISDatabase` natively, this file shrinks and
eventually dies — do not grow features on it.

Pure Python + numpy + mvis.core.types. NO sqlite, NO io/detection imports
(the geometry object is duck-typed: anything with point_3d()/num_points()).
"""
from __future__ import annotations

import numpy as np

from mvis.mvisdb.mvisdatabase import (
    Board, BoardPt, BoardType, CalibChoice, CameraInfo, CameraStream, Corner,
    MVISDatabase, Rig, Run, TagDetection,
)

__all__ = ["board_from_geometry", "db_from_frame_observations", "arrays_to_legacy",
           "db_from_legacy_arrays", "frame_observations_from_db"]

_TYPE_BY_CLASSNAME = {
    "AprilTagBoard": BoardType.APRILGRID,   # the REAL class name (capital T) —
    "ApriltagBoard": BoardType.APRILGRID,   # the misspelling let every real
    "ArucoBoard": BoardType.ARUCO_GRID,     # AprilGrid fall back to the legacy
    "Chessboard": BoardType.CHESSBOARD,     # loop silently; caught by review,
    "ChessBoard": BoardType.CHESSBOARD,     # pinned by test now
    # ChArUco points are chessboard corners, one per id: stored as chessboard rows (the
    # schema's board_type CHECK has no 'charuco'; the ChArUco layout stays in the config
    # and the cache key)
    "CharucoBoard": BoardType.CHESSBOARD,
}


def board_from_geometry(geom, *, board_id: int = 0,
                        board_type: BoardType | None = None,
                        corners_per_tag: int | None = None) -> Board:
    """Duck-typed adapter: any object with point_3d(pid) and num_points().

    board_type/corners_per_tag are inferred from the class name when not given;
    pass them explicitly for stubs. rows/cols/spacing/size are best-effort from
    private attrs — the bridge only needs GEOMETRY to be exact, and it is:
    every board_pt_3d comes verbatim from geom.point_3d.
    """
    if board_type is None:
        board_type = _TYPE_BY_CLASSNAME.get(type(geom).__name__)
        if board_type is None:
            raise ValueError(
                f"cannot infer board_type from {type(geom).__name__}; pass board_type=")
    cpt = corners_per_tag if corners_per_tag is not None else (
        1 if board_type is BoardType.CHESSBOARD else 4)
    n = int(geom.num_points())
    pts = {}
    for pid in range(n):
        corner = None if cpt == 1 else Corner(pid % cpt)
        pts[pid] = BoardPt(pid, pid // cpt, corner,
                           np.asarray(geom.point_3d(pid), float))
    return Board(
        board_id=board_id, board_type=board_type,
        rows=int(getattr(geom, "_rows", 0) or 0),
        cols=int(getattr(geom, "_cols", 0) or 0),
        spacing=float(getattr(geom, "_spacing", 0.0) or 0.0),
        size=float(getattr(geom, "_tag_size", getattr(geom, "_square_size", 0.0)) or 0.0),
        start_id=int(getattr(geom, "_start_id", 0) or 0),
        pts=pts)


def db_from_frame_observations(all_frames: dict[int, list],
                               geom, *, name: str = "bridge",
                               image_sizes: dict[int, tuple[int, int]] | None = None,
                               cam_names: dict[int, str] | None = None,
                               board_id: int = 0,
                               board_type: BoardType | None = None,
                               corners_per_tag: int | None = None) -> MVISDatabase:
    """{cam_id: [FrameObservations]} + a board geometry -> MVISDatabase.

    Detections are copied VERBATIM: uv float64 untouched, point_id as emitted,
    detector_emission_order = the index the observation had in its frame's list
    (the bit-identity replay column). Frames keep their input order per camera;
    the structure's validate() then enforces time order, same as ingest will.
    """
    db = MVISDatabase(name=name, rig=Rig(base_camera_id=min(all_frames) if all_frames else 0),
                      run=Run(CalibChoice.CAM_ONLY))
    db.boards[board_id] = board_from_geometry(
        geom, board_id=board_id, board_type=board_type, corners_per_tag=corners_per_tag)
    for cam_id in sorted(all_frames):
        frames = all_frames[cam_id]
        info = CameraInfo(name=(cam_names or {}).get(cam_id, f"cam{cam_id}"),
                          resolution=(image_sizes or {}).get(cam_id))
        db.streams.append(CameraStream(cam_id, info))
        for fo in frames:
            f = db.add_frame(cam_id, fo.timestamp)
            dets = [TagDetection(uv=np.asarray(o.pixel, float),
                                 point_id=int(o.point_id),
                                 detector_emission_order=k)
                    for k, o in enumerate(fo.observations)]
            db.add_board_detection(f, board_id, dets)
    return db


def db_from_legacy_arrays(per_cam: dict[int, tuple], geom, *,
                          name: str = "bridge",
                          image_sizes: dict[int, tuple[int, int]] | None = None,
                          board_id: int = 0,
                          board_type: BoardType | None = None,
                          corners_per_tag: int | None = None) -> MVISDatabase:
    """{cam_id: (t_cam, ids_list, uv_list)} -> MVISDatabase — the inverse of
    arrays_to_legacy. Covers BOTH producers of the ragged shape: the simulator
    (CamStream.times/ids/uv) and the cam-IMU loader. Emission order = index
    within each frame's ids array; uv copied verbatim."""
    db = MVISDatabase(name=name,
                      rig=Rig(base_camera_id=min(per_cam) if per_cam else 0),
                      run=Run(CalibChoice.CAM_ONLY))
    db.boards[board_id] = board_from_geometry(
        geom, board_id=board_id, board_type=board_type, corners_per_tag=corners_per_tag)
    for cam_id in sorted(per_cam):
        t_cam, ids_list, uv_list = per_cam[cam_id]
        info = CameraInfo(name=f"cam{cam_id}",
                          resolution=(image_sizes or {}).get(cam_id))
        db.streams.append(CameraStream(cam_id, info))
        for t, ids, uv in zip(np.asarray(t_cam, float), ids_list, uv_list):
            f = db.add_frame(cam_id, float(t))
            uv = np.asarray(uv, float).reshape(-1, 2)
            dets = [TagDetection(uv=uv[k], point_id=int(pid),
                                 detector_emission_order=k)
                    for k, pid in enumerate(np.asarray(ids, int))]
            db.add_board_detection(f, board_id, dets)
    return db


def frame_observations_from_db(db: MVISDatabase, camera_id: int, *,
                               selected_only: bool = False,
                               valid_only: bool = False) -> list:
    """MVISDatabase -> list[FrameObservations] — the cam-only pipelines' shape.

    Observations come back in detector emission order with uv float64 verbatim,
    so a solver fed these sees LITERALLY the same inputs as one fed the
    originals (tests/mvisdb/test_mvisdb_solver_path.py proves it end to end).
    Imported lazily so the structure itself stays free of mvis.core at import
    of arrays-only users."""
    from mvis.core.types import FrameObservations, Observation
    out = []
    stream = db.camera(camera_id)
    for f in stream.frames:
        if selected_only and not f.is_selected_to_use:
            continue
        obs = [Observation(point_id=d.point_id, pixel=d.uv.copy(), row=d.row)
               for g in f.board_detections
               for d in sorted(g.detections, key=lambda d: d.detector_emission_order)
               if not (valid_only and not d.is_valid)
               and not (selected_only and not d.is_selected)]
        out.append(FrameObservations(camera_id=camera_id, timestamp=f.timestamp,
                                     observations=obs,
                                     image_size=stream.info.resolution))
    return out


def arrays_to_legacy(db: MVISDatabase, camera_id: int, *,
                     selected_only: bool = False, valid_only: bool = False):
    """MVISDatabase -> the cam-IMU pipeline's ragged (t_cam, ids_list, uv_list).

    Defaults reproduce today's contract exactly (no flag filtering — legacy
    callers did their own), in detector emission order, matching
    io.frame_detector.frames_to_arrays bit for bit.
    """
    t_cam, ids_list, uv_list = [], [], []
    for f in db.camera(camera_id).frames:
        if selected_only and not f.is_selected_to_use:
            continue
        dets = [d for g in f.board_detections for d in sorted(
                    g.detections, key=lambda d: d.detector_emission_order)
                if not (valid_only and not d.is_valid)
                and not (selected_only and not d.is_selected)]
        t_cam.append(f.timestamp)
        ids_list.append(np.array([d.point_id for d in dets], int))
        uv_list.append(np.array([d.uv for d in dets], float).reshape(-1, 2))
    return np.asarray(t_cam, float), ids_list, uv_list
