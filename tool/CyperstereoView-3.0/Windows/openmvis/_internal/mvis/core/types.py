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
"""Core data types for camera calibration."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Observation:
    """A single detected board feature in an image."""

    point_id: int  # board corner/tag corner ID
    pixel: np.ndarray  # [u, v] detected image coordinate
    timestamp: float = 0.0  # capture time (seconds)
    row: int = 0  # pixel row (for rolling shutter)

    def __post_init__(self):
        """Normalize `pixel` to a float64 ndarray."""
        self.pixel = np.asarray(self.pixel, dtype=np.float64)


@dataclass
class FrameObservations:
    """All detected board features in a single frame from one camera."""

    camera_id: int
    timestamp: float
    observations: list[Observation]
    image_size: tuple[int, int] = (0, 0)  # (width, height)

    @property
    def num_points(self) -> int:
        """Number of detected features in this frame."""
        return len(self.observations)

    @property
    def point_ids(self) -> np.ndarray:
        """Board point IDs of the detections, aligned with `pixels` rows."""
        return np.array([obs.point_id for obs in self.observations])

    @property
    def pixels(self) -> np.ndarray:
        """Return Nx2 array of detected pixel coordinates."""
        return np.array([obs.pixel for obs in self.observations])


@dataclass
class CameraParams:
    """Calibrated camera parameters."""

    camera_id: int
    intrinsics: np.ndarray  # [fx, fy, cx, cy]
    distortion: np.ndarray  # model coeffs (core.camera_models.DIST_NAMES), e.g. [k1, k2, p1, p2]
    distortion_model: str  # a core.camera_models.MODELS name
    image_size: tuple[int, int] = (0, 0)  # (width, height)
    rolling_shutter_readout: float = 0.0  # seconds, 0 = global shutter
    # Formal 1-sigma of the readout from the solve's marginal covariance (seconds);
    # None for global-shutter cameras or when the marginals were not computed.
    rolling_shutter_readout_sigma: float | None = None

    def __post_init__(self):
        """Normalize `intrinsics`/`distortion` to float64 ndarrays."""
        self.intrinsics = np.asarray(self.intrinsics, dtype=np.float64)
        self.distortion = np.asarray(self.distortion, dtype=np.float64)

    @property
    def fx(self) -> float:
        """Focal length x (px)."""
        return float(self.intrinsics[0])

    @property
    def fy(self) -> float:
        """Focal length y (px)."""
        return float(self.intrinsics[1])

    @property
    def cx(self) -> float:
        """Principal point x (px)."""
        return float(self.intrinsics[2])

    @property
    def cy(self) -> float:
        """Principal point y (px)."""
        return float(self.intrinsics[3])


@dataclass
class ExtrinsicTransform:
    """Extrinsic transform from one camera to the reference camera."""

    from_camera: int
    to_camera: int  # reference camera
    T_to_from: np.ndarray  # 4x4 SE(3) transformation matrix
    so3: np.ndarray  # rotation vector (3,) in radians
    translation: np.ndarray  # translation (3,) in meters

    def __post_init__(self):
        """Normalize the matrix/vector fields to float64 ndarrays."""
        self.T_to_from = np.asarray(self.T_to_from, dtype=np.float64)
        self.so3 = np.asarray(self.so3, dtype=np.float64)
        self.translation = np.asarray(self.translation, dtype=np.float64)


@dataclass
class ReprojectionErrorStats:
    """Reprojection error statistics for one camera."""

    camera_id: int
    rms: float
    mean: float
    median: float
    max: float = 0.0
    num_observations: int = 0
    p90: float = 0.0
    p95: float = 0.0
    std: float = 0.0                 # population std of the corner norms, as Kalibr prints it
    inlier: dict | None = None       # reproj_stats.inlier_stats: the same corners within 3 x pixel sigma


@dataclass
class CalibrationResult:
    """Complete calibration result."""

    cameras: list[CameraParams]
    extrinsics: list[ExtrinsicTransform] = field(default_factory=list)
    reference_camera: int = 0
    time_offsets: dict[int, float] = field(default_factory=dict)
    # Observations within this many seconds of a ref pose were projected at it
    # (camera_projection.synced_time); 0 = at their own stamps. Re-projections
    # of this result (outlier rejection, reports) use the same rule.
    snap_sync_tolerance_s: float = 0.0
    board_poses: list[np.ndarray] = field(default_factory=list)  # list of 4x4 SE(3)
    board_poses_gtsam: list = field(default_factory=list)  # list of gtsam.Pose3
    board_pose_times: list[float] = field(default_factory=list)  # ref-cam timeline timestamps
    # Reference-camera indices relative to the exact frame list passed to the
    # Optimizer.  Parallel to board_poses(_gtsam) and board_pose_times.
    valid_frame_indices: list[int] = field(default_factory=list)
    board_points_refined: Optional[np.ndarray] = None  # Nx3 refined board points
    board_heights: Optional[np.ndarray] = None  # N out-of-plane corner heights (m), plane removed
    # N vision factors per corner in the solve that estimated board_heights; 0 = the height was
    # never constrained (e.g. a corner dropped by corner_subsample) and sits at its prior
    board_height_n_obs: Optional[np.ndarray] = None
    board_point_bias: Optional[np.ndarray] = None  # Nx3 [mean du, mean dv (px), n_obs] per corner
    # {camera_id: count of each corner's fate at the final poses}, see
    # calibration.corner_accounting; a diagnostic, absent from older results
    corner_accounting: Optional[dict] = None
    # {camera_id: timestamps of the frames the solve gave factors}
    factor_frames: Optional[dict] = None
    reprojection_errors: list[ReprojectionErrorStats] = field(default_factory=list)
    overall_rms: float = 0.0
    # Hierarchical wall-clock timings produced by Timer.to_dict().
    # Optional; absent for results loaded from older YAML files.
    timings: Optional[dict] = None
    # The LM rescue report of the solve that made this result (calibration.lm_rescue): None unless LM
    # stopped stuck or out of iterations far from its minimum; `failed` = still stuck after the rescue.
    lm_rescue: Optional[dict] = None
    # Diverged pose removal stats from optimizer sanitizer.
    # {"total": N, "removed": M, "kept": K}
    diverged_pose_stats: Optional[dict] = None
    # Motion excitation analysis: per-camera metrics dict.
    motion_excitation: Optional[dict] = None
