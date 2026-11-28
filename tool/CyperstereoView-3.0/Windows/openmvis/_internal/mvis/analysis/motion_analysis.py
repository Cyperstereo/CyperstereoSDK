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
"""Motion excitation analysis for calibration quality assessment.

Computes metrics from optimized board poses to determine if the
calibration data has sufficient motion diversity for each parameter.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from mvis.calibration.factors.lie_groups import log_so3

logger = logging.getLogger(__name__)


@dataclass
class MotionMetrics:
    """Motion excitation metrics for one camera."""

    camera_id: int = 0
    n_poses: int = 0

    # Translation (camera position in board frame)
    x_range: float = 0.0   # meters
    y_range: float = 0.0
    z_range: float = 0.0
    z_min: float = 0.0
    z_max: float = 0.0
    z_ratio: float = 1.0   # z_max / z_min
    volume: float = 0.0    # x_range * y_range * z_range (m³)

    # Rotation diversity (board frame)
    rot_angle_min: float = 0.0   # degrees from identity
    rot_angle_max: float = 0.0
    rot_angle_std: float = 0.0
    axis_eigenvalues: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    axis_ratio: float = 0.0  # λ_min / λ_max (1=isotropic, 0=single-axis)

    # Angular velocity (camera frame)
    omega_median: float = 0.0  # rad/s
    omega_max: float = 0.0
    omega_std: float = 0.0

    # RS excitation: peak lateral angular velocity in camera frame.
    # omega_x (roll) and omega_y (pitch) create row-dependent displacement.
    # omega_z (yaw) creates minimal RS signal (radial, not row-dependent).
    rs_excitation: float = 0.0      # max sqrt(omega_x² + omega_y²) in cam frame
    rs_excitation_median: float = 0.0

    # Feature coverage (set externally from coverage plot)
    hull_coverage_pct: float = 0.0  # convex hull area / image area %

    def to_dict(self) -> dict:
        """Serialize for YAML output."""
        return {
            "camera_id": self.camera_id,
            "n_poses": self.n_poses,
            "translation": {
                "x_range": round(self.x_range, 4),
                "y_range": round(self.y_range, 4),
                "z_range": round(self.z_range, 4),
                "z_min": round(self.z_min, 4),
                "z_max": round(self.z_max, 4),
                "z_ratio": round(self.z_ratio, 2),
                "volume": round(self.volume, 6),
            },
            "rotation": {
                "angle_min_deg": round(self.rot_angle_min, 1),
                "angle_max_deg": round(self.rot_angle_max, 1),
                "angle_std_deg": round(self.rot_angle_std, 1),
                "axis_eigenvalues": [round(e, 6) for e in self.axis_eigenvalues],
                "axis_ratio": round(self.axis_ratio, 3),
            },
            "angular_velocity": {
                "omega_median": round(self.omega_median, 3),
                "omega_max": round(self.omega_max, 3),
                "omega_std": round(self.omega_std, 3),
                "rs_excitation": round(self.rs_excitation, 3),
                "rs_excitation_median": round(self.rs_excitation_median, 3),
            },
        }


#: consecutive board poses closer than this in time are one pose (duplicate stamp)
_MIN_PAIR_DT = 1e-6


def valid_board_poses(board_poses: list, board_pose_times: list[float]) -> list:
    """(4x4 T_board_cam, t) of every solved pose: None and identity (unsolved) skipped."""
    valid = []
    for i, bp in enumerate(board_poses):
        if bp is None:
            continue
        bp = np.asarray(bp)
        if np.allclose(bp, np.eye(4)):
            continue
        valid.append((bp, board_pose_times[i] if i < len(board_pose_times) else 0.0))
    return valid


def motion_series(valid: list) -> list:
    """Per consecutive pair of `valid_board_poses` (the one dt rule: pairs under
    `_MIN_PAIR_DT` apart are skipped): (t_j, dt, rvec_board, omega_cam), with
    rvec_board = Log(R_j R_i^T) and omega_cam the camera-frame angular rate."""
    out = []
    for (bp_i, t_i), (bp_j, t_j) in zip(valid, valid[1:]):
        dt = t_j - t_i
        if dt < _MIN_PAIR_DT:
            continue
        rvec = log_so3(bp_j[:3, :3] @ bp_i[:3, :3].T)
        R_rel_cam = np.linalg.inv(bp_j)[:3, :3] @ np.linalg.inv(bp_i)[:3, :3].T
        out.append((t_j, dt, rvec, log_so3(R_rel_cam) / dt))
    return out


def hull_coverage(pts, width: float, height: float):
    """(convex-hull area of pixel points as % of the image, closed hull polygon).
    Fewer than 3 points or a degenerate set give (0.0, None)."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    if len(pts) < 3:
        return 0.0, None
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
    except Exception:                         # collinear / scipy missing
        return 0.0, None
    poly = pts[hull.vertices]
    return hull.volume / (width * height) * 100, np.vstack([poly, poly[:1]])  # 2D: volume = area


def compute_motion_metrics(
    board_poses: list,
    board_pose_times: list[float],
    camera_id: int = 0,
) -> MotionMetrics:
    """Compute motion excitation metrics from optimized board poses.

    Args:
        board_poses: List of 4×4 T_board_cam matrices (or None).
        board_pose_times: Parallel timestamps.
        camera_id: Camera ID for labeling.

    Returns:
        MotionMetrics with all computed values; `_axis_vectors` holds the
        non-zero relative rotation vectors (axis-spread plot).
    """
    m = MotionMetrics(camera_id=camera_id)
    valid = valid_board_poses(board_poses, board_pose_times)
    m.n_poses = len(valid)
    if m.n_poses < 2:
        return m

    # === Translation analysis (camera position in board frame) ===
    # Camera position in board frame = inv(T_board_cam)[:3,3]
    positions = np.array([np.linalg.inv(bp)[:3, 3] for bp, _t in valid])
    m.x_range = float(positions[:, 0].max() - positions[:, 0].min())
    m.y_range = float(positions[:, 1].max() - positions[:, 1].min())
    m.z_range = float(positions[:, 2].max() - positions[:, 2].min())
    m.z_min = float(positions[:, 2].min())
    m.z_max = float(positions[:, 2].max())
    m.z_ratio = m.z_max / max(m.z_min, 1e-6) if m.z_min > 0.01 else 1.0
    m.volume = m.x_range * m.y_range * m.z_range

    series = motion_series(valid)

    # === Rotation analysis (inter-frame angular changes) ===
    # How much the board orientation VARIES between frames, not the absolute
    # angle from identity (dominated by the fixed camera-board setup).
    if series:
        inter_angles = np.array([float(np.linalg.norm(r)) * 180.0 / np.pi
                                 for _t, _dt, r, _w in series])
        m.rot_angle_min = float(inter_angles.min())
        m.rot_angle_max = float(inter_angles.max())
        m.rot_angle_std = float(inter_angles.std())

    # === Rotation axis spread (board frame) ===
    axis_angle_vectors = [r for _t, _dt, r, _w in series if np.linalg.norm(r) > 1e-6]
    m._axis_vectors = axis_angle_vectors or None
    if len(axis_angle_vectors) >= 3:
        aa = np.array(axis_angle_vectors)
        cov = np.cov(aa.T)
        eigvals = sorted(np.linalg.eigvalsh(cov), reverse=True)
        m.axis_eigenvalues = [float(e) for e in eigvals]
        m.axis_ratio = eigvals[2] / max(eigvals[0], 1e-12)

    # === Angular velocity ===
    # Board frame: ||omega|| for total speed (time offset signal).
    # Camera frame: omega_x, omega_y for RS excitation (row-dependent
    # displacement); omega_z (yaw) creates radial motion, a weak RS signal.
    if series:
        omega_norms = np.array([float(np.linalg.norm(r / dt)) for _t, dt, r, _w in series])
        m.omega_median = float(np.median(omega_norms))
        m.omega_max = float(np.max(omega_norms))
        m.omega_std = float(np.std(omega_norms))
        omega_perps = np.array([float(np.sqrt(w[0]**2 + w[1]**2)) for _t, _dt, _r, w in series])
        m.rs_excitation = float(np.max(omega_perps))
        m.rs_excitation_median = float(np.median(omega_perps))

    return m


def assess_motion_quality(
    metrics: MotionMetrics,
    calibrate_rs: bool = False,
    calibrate_dt: bool = False,
) -> dict[str, str]:
    """Return per-parameter quality grade.

    Returns dict: parameter_name → "good" / "marginal" / "poor".

    NOTE: All thresholds are empirical recommendations, not rigorously
    derived. They should be validated against benchmark data.
    """
    grades = {}

    # Intrinsics (fx): z_ratio
    if metrics.z_ratio > 2.0:
        grades["intrinsics_fx"] = "good"
    elif metrics.z_ratio > 1.5:
        grades["intrinsics_fx"] = "marginal"
    else:
        grades["intrinsics_fx"] = "poor"

    # Distortion: hull_coverage
    if metrics.hull_coverage_pct > 60:
        grades["distortion"] = "good"
    elif metrics.hull_coverage_pct > 30:
        grades["distortion"] = "marginal"
    else:
        grades["distortion"] = "poor"

    # Rotation diversity: axis_ratio
    if metrics.axis_ratio > 0.1:
        grades["rotation_diversity"] = "good"
    elif metrics.axis_ratio > 0.03:
        grades["rotation_diversity"] = "marginal"
    else:
        grades["rotation_diversity"] = "poor"

    # Time offset: omega_std
    if calibrate_dt:
        if metrics.omega_std > 0.5:
            grades["time_offset"] = "good"
        elif metrics.omega_std > 0.2:
            grades["time_offset"] = "marginal"
        else:
            grades["time_offset"] = "poor"

    # RS readout: rs_excitation
    if calibrate_rs:
        if metrics.rs_excitation > 1.0:
            grades["rs_readout"] = "good"
        elif metrics.rs_excitation > 0.3:
            grades["rs_readout"] = "marginal"
        else:
            grades["rs_readout"] = "poor"

    return grades
