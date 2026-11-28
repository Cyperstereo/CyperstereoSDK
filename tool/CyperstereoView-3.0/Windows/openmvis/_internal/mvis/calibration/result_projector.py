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
"""The per-camera projection of a solved camera-only `CalibrationResult`.

Every camera-only consumer that reprojects board corners after a solve (the outlier filter between
passes, the HTML report's residuals, the corner accounting) builds its projection here, so they all
score a corner against the board the solve fitted: `result.board_points_refined` when the solve
estimated the board shape, the board's own nominal points otherwise. The forward model is
`camera_projection.project_observation`, the one the optimizer's factor minimizes.

A result without `board_points_refined` projects exactly as the walkers did before: the nominal
point comes from `board.point_3d`, call for call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gtsam
import numpy as np

from mvis.calibration.factors.camera_projection import (
    _resolve_distortion_model, find_bracketing_poses, k_cal_vec, observation_pose,
    project_at_pose, synced_time,
)


@dataclass(frozen=True)
class CameraProjection:
    """One camera's fixed projection inputs from a result."""
    camera_id: int
    model: str
    cal_vec: np.ndarray
    is_ref: bool
    T_ext: Optional[gtsam.Pose3]      # T_ref_cam, None for the reference camera
    td: float                         # time offset (s), 0 for the reference camera
    tr: float                         # rolling-shutter readout (s)
    image_height: float


def board_pose_timeline(result):
    """Ref-camera pose timeline of a result, for report and residual lookups: (indices, times, {index:
    Pose3}). None and identity poses (from the diverged-pose sanitizer) are left out, so bracketing
    only uses valid poses."""
    _valid_bp = [
        (i, t, p) for i, (t, p) in enumerate(
            zip(result.board_pose_times, result.board_poses))
        if p is not None and not np.allclose(p, np.eye(4))
    ]
    if _valid_bp:
        return ([i for i, _, _ in _valid_bp],
                np.array([t for _, t, _ in _valid_bp], dtype=np.float64),
                {i: gtsam.Pose3(p) for i, _, p in _valid_bp})
    return [], np.array([], dtype=np.float64), {}


class ResultProjector:
    """Projects board corners with a solved result's cameras, poses and board.

    `board_point(pid)` is the refined board point when the result carries `board_points_refined`,
    the nominal `board.point_3d(pid)` otherwise. `camera(cid)` gives the camera's projection inputs
    (K vector, extrinsic, time offset, readout, image height). `bracket(t)` finds a frame's two
    bracketing ref-camera poses on the result's valid timeline, and `project` / `residual` project
    one observation at such a bracket."""

    def __init__(self, result, board, image_sizes: Optional[dict] = None):
        self.result, self.board = result, board
        self.image_sizes = image_sizes or {}
        refined = getattr(result, "board_points_refined", None)
        self.board_points = None if refined is None else np.asarray(refined, dtype=np.float64)
        self._cams = {c.camera_id: c for c in result.cameras}
        self._ext = {e.from_camera: gtsam.Pose3(e.T_to_from) for e in result.extrinsics}
        self._proj: dict[int, CameraProjection] = {}
        self._timeline = None

    @property
    def uses_refined_board(self) -> bool:
        """Whether corners are scored against the solve's refined board."""
        return self.board_points is not None

    def board_point(self, point_id: int) -> np.ndarray:
        """The board-frame point a corner is scored against."""
        if self.board_points is not None:
            return self.board_points[point_id]
        return self.board.point_3d(point_id)

    def has_camera(self, camera_id) -> bool:
        return camera_id in self._cams

    def camera(self, camera_id) -> CameraProjection:
        """This camera's projection inputs (cached)."""
        p = self._proj.get(camera_id)
        if p is None:
            cam = self._cams[camera_id]
            is_ref = camera_id == self.result.reference_camera
            cal_vec = k_cal_vec(cam.distortion_model, cam.intrinsics, cam.distortion)
            p = CameraProjection(
                camera_id=camera_id,
                model=_resolve_distortion_model(None, cam.distortion_model, cal_vec),
                cal_vec=cal_vec,
                is_ref=is_ref,
                T_ext=None if is_ref else self._ext.get(camera_id),
                td=0.0 if is_ref else float(self.result.time_offsets.get(camera_id, 0.0)),
                tr=float(cam.rolling_shutter_readout),
                image_height=self.image_sizes.get(camera_id, (640, 480))[1])
            self._proj[camera_id] = p
        return p

    def timeline(self):
        """`board_pose_timeline(result)`, computed once."""
        if self._timeline is None:
            self._timeline = board_pose_timeline(self.result)
        return self._timeline

    def bracket(self, timestamp: float):
        """(t, bp0, bp1, t0, t1) for a frame stamp on the valid ref-camera timeline, or None."""
        idx, times, poses = self.timeline()
        if len(times) < 2:
            return None
        t = synced_time(timestamp, times, getattr(self.result, "snap_sync_tolerance_s", 0.0))
        b = find_bracketing_poses(t, times, idx)
        if b is None:
            return None
        i0, i1, t0, t1 = b
        if i0 not in poses or i1 not in poses:
            return None
        return t, poses[i0], poses[i1], t0, t1

    def solved_bracket(self, cam: CameraProjection, timestamp: float):
        """`bracket(timestamp)`, or None when the solve had no pose for this frame: a ref-camera
        frame always sits on its own pose node, and with no time offset estimated (snap tolerance
        > 0) a non-ref frame sits on the ref pose it snapped to. A frame off every pose node was
        dropped by the solve (a gate, or an unpaired co-frame); interpolating it between ref poses
        1/target_hz apart scores a pose the solve never fitted (pixels off for a moving board)."""
        b = self.bracket(timestamp)
        if b is None:
            return None
        interpolates = not cam.is_ref and getattr(self.result, "snap_sync_tolerance_s", 0.0) <= 0
        t, _, _, t0, t1 = b
        on_node = abs(t - t0) < 1e-9 or abs(t - t1) < 1e-9
        return b if interpolates or on_node else None

    def observation_pose(self, cam: CameraProjection, obs, bracket) -> gtsam.Pose3:
        """The camera pose T_board_cam at which `obs` is projected, at `bracket` = (t, bp0, bp1,
        t0, t1) from `bracket()`."""
        t, bp0, bp1, t0, t1 = bracket
        return observation_pose(
            bp0=bp0, bp1=bp1, time_meas=t, time_0=t0, time_1=t1,
            T_ext=cam.T_ext, td=cam.td, tr=cam.tr,
            pixel_row=obs.row, image_height=cam.image_height)

    def project(self, cam: CameraProjection, obs, bracket,
                cal_vec: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """(2,) projected pixel of `obs` at `bracket`, or None if the projection fails. `cal_vec`
        overrides the camera's K vector."""
        return project_at_pose(self.observation_pose(cam, obs, bracket),
                               self.board_point(obs.point_id),
                               cam.cal_vec if cal_vec is None else cal_vec, cam.model)

    def residual(self, cam: CameraProjection, obs, bracket) -> Optional[float]:
        """Pixel residual norm |proj - obs.pixel| at `bracket`, or None if the projection fails."""
        proj = self.project(cam, obs, bracket)
        return None if proj is None else float(np.linalg.norm(proj - obs.pixel))
