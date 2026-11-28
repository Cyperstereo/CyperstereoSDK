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
"""Write calibration results to JSON or YAML.

Two output modes:

1. **Calibration-only** (``write_calibration_yaml``): minimal, deployable.
   Contains intrinsics, distortion, extrinsics, time offsets, rolling
   shutter readout, and summary reprojection errors. This is what you
   ship to a robot or SLAM system.

2. **Full replay** (``write_full_yaml``): everything needed to
   reconstruct the calibration visualization in Rerun or regenerate
   HTML reports without re-running calibration. Includes board poses
   (per-frame SE(3) trajectory), board 3D corner positions, timing
   breakdown, filter stats, and config metadata.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from mvis.core.types import CalibrationResult
from mvis.io.result_schema import camera_intrinsics_block


def _ndarray_to_list(obj):
    """Recursively convert numpy arrays to lists for serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    return obj


class OutputWriter:
    """Write calibration results to JSON or YAML files."""

    # ------------------------------------------------------------------
    # Calibration-only dict (File 1: deploy to robot/SLAM)
    # ------------------------------------------------------------------

    @staticmethod
    def to_calibration_dict(result: CalibrationResult) -> dict:
        """Minimal calibration data — intrinsics, distortion, extrinsics,
        time offsets, rolling shutter, and summary error stats."""
        data = {
            "cameras": [],
            "extrinsics": {
                "reference_camera": result.reference_camera,
                "transforms": [],
            },
            "time_offsets": {
                "reference_camera": result.reference_camera,
                "offsets": [],
            },
            "reprojection_errors": {
                "per_camera": [],
                "overall_rms": float(result.overall_rms),
            },
        }

        for cam in result.cameras:
            data["cameras"].append(camera_intrinsics_block(
                cam.camera_id, cam.distortion_model, cam.intrinsics,
                cam.distortion, cam.image_size, cam.rolling_shutter_readout))
            sig = getattr(cam, "rolling_shutter_readout_sigma", None)
            if sig is not None:
                # formal 1-sigma of the readout (s) from the solve's marginals
                data["cameras"][-1]["rolling_shutter_readout_sigma"] = float(sig)

        for ext in result.extrinsics:
            data["extrinsics"]["transforms"].append({
                "from": ext.from_camera,
                "to": ext.to_camera,
                "T_to_from": ext.T_to_from.tolist()
                    if isinstance(ext.T_to_from, np.ndarray)
                    else ext.T_to_from,
                "so3": ext.so3.tolist()
                    if isinstance(ext.so3, np.ndarray)
                    else ext.so3,
                "translation": ext.translation.tolist()
                    if isinstance(ext.translation, np.ndarray)
                    else ext.translation,
            })

        for cam_id, dt in result.time_offsets.items():
            data["time_offsets"]["offsets"].append({
                "camera": cam_id,
                "dt": float(dt),
            })

        for err in result.reprojection_errors:
            data["reprojection_errors"]["per_camera"].append({
                "camera": err.camera_id,
                "rms": float(err.rms),
                "mean": float(err.mean),
                "median": float(err.median),
                "std": float(err.std),
                "p90": float(err.p90),
                "p95": float(err.p95),
                "max": float(err.max),
                "num_observations": int(err.num_observations),
                **({"inlier": {k: (int(v) if k.startswith("n_") else float(v))
                               for k, v in err.inlier.items()
                               if k in ("mean", "median", "std", "p90", "p95", "max", "n_corners", "n_excluded",
                                        "threshold_px")}}
                   if err.inlier else {}),
            })

        # Board shape: per-corner heights when estimated,
        # and the per-corner mean-residual diagnostic (board-shape / detection bias)
        if getattr(result, "board_heights", None) is not None:
            h = np.asarray(result.board_heights, float)
            n_obs = getattr(result, "board_height_n_obs", None)
            seen = np.asarray(n_obs) > 0 if n_obs is not None else np.ones(len(h), bool)
            hs = h[seen] if seen.any() else h
            data["board_shape"] = {
                "model": "pose + per-corner height",
                "heights_m": [float(v) for v in h],
                "rms_mm": float(np.std(hs) * 1e3),            # over the corners the solve saw
                "peak_to_peak_mm": float((hs.max() - hs.min()) * 1e3),
            }
            if n_obs is not None:
                data["board_shape"]["n_obs"] = [int(v) for v in n_obs]
        if getattr(result, "board_point_bias", None) is not None:
            b = np.asarray(result.board_point_bias, float)
            seen = b[:, 2] > 0
            mag = np.hypot(b[seen, 0], b[seen, 1]) if seen.any() else np.zeros(0)
            data["board_point_bias"] = {
                "median_px": float(np.median(mag)) if mag.size else 0.0,
                "max_px": float(mag.max()) if mag.size else 0.0,
                "n_points_seen": int(seen.sum()),
            }
        accounting = getattr(result, "corner_accounting", None)
        if accounting:
            data["corner_accounting"] = {
                "per_camera": [accounting[c] for c in sorted(accounting)]}

        return data

    # ------------------------------------------------------------------
    # Full replay dict (File 2: Rerun replay + report regen)
    # ------------------------------------------------------------------

    @staticmethod
    def to_full_dict(
        result: CalibrationResult,
        board=None,
        corner_subsample: str = "all",
        config_path: str = "",
        detection_filter_stats: dict | None = None,
    ) -> dict:
        """Full diagnostic dump — calibration + board poses + board 3D +
        timing + filter stats + config metadata."""
        data = OutputWriter.to_calibration_dict(result)

        # Board poses: per-frame SE(3) trajectory for the ref camera.
        board_poses_list = []
        for i, pose in enumerate(result.board_poses):
            if pose is None:
                continue
            ts = (result.board_pose_times[i]
                  if i < len(result.board_pose_times) else 0.0)
            T = pose.tolist() if isinstance(pose, np.ndarray) else pose
            board_poses_list.append({
                "frame_index": i,
                "timestamp": float(ts),
                "T_board_cam": T,
            })
        data["board_poses"] = board_poses_list

        # Board 3D corner positions.
        if board is not None:
            pts = board.all_points_3d()
            data["board"] = {
                "type": board.__class__.__name__,
                "num_points": int(board.num_points()),
                "num_tags": int(getattr(board, "_num_tags", 0)),
                "rows": int(getattr(board, "_rows", 0)),
                "cols": int(getattr(board, "_cols", 0)),
                "tag_size": float(getattr(board, "_tag_size", 0.0)),
                "points_3d": pts.tolist()
                    if isinstance(pts, np.ndarray) else pts,
            }
            if hasattr(board, "_charuco_marker"):     # ChArUco: enough to rebuild the board
                data["board"].update(
                    squares_rows=int(board._squares_rows), squares_cols=int(board._squares_cols),
                    square_size=float(board._square_size), marker_size=float(board._charuco_marker),
                    aruco_dict=board._aruco_dict_name, charuco_legacy=bool(board._legacy_pattern),
                    charuco_refine=bool(board._refine_corners))

        # Corner subsample mode used.
        data["corner_subsample"] = corner_subsample

        # Config path for provenance.
        if config_path:
            data["config_path"] = config_path

        # Hierarchical wall-clock timings.
        timings = getattr(result, "timings", None)
        if timings:
            data["timings"] = timings

        # LM rescue report (calibration.lm_rescue); `failed` makes `mvis verdict` grade the result RED.
        lm_rescue = getattr(result, "lm_rescue", None)
        if lm_rescue:
            data["lm_rescue"] = lm_rescue

        # Diverged pose removal stats.
        dp_stats = getattr(result, "diverged_pose_stats", None)
        if dp_stats:
            data["diverged_pose_stats"] = dp_stats

        # Motion excitation metrics.
        motion = getattr(result, "motion_excitation", None)
        if motion:
            data["motion_excitation"] = motion

        # Detection-quality filter stats.
        if detection_filter_stats:
            data["detection_filter_stats"] = detection_filter_stats

        return data

    # ------------------------------------------------------------------
    # Back-compat: to_dict = calibration-only (existing callers)
    # ------------------------------------------------------------------

    @staticmethod
    def to_dict(result: CalibrationResult) -> dict:
        """Back-compat alias for ``to_calibration_dict``.

        Also includes ``timings`` if present (for the regression
        dashboard's ``collect_stats`` which reads the main YAML).
        """
        data = OutputWriter.to_calibration_dict(result)
        timings = getattr(result, "timings", None)
        if timings:
            data["timings"] = timings
        return data

    # ------------------------------------------------------------------
    # Write methods
    # ------------------------------------------------------------------

    @staticmethod
    def write_calibration_yaml(result: CalibrationResult, path: str) -> None:
        """Write calibration-only YAML (File 1)."""
        data = OutputWriter.to_calibration_dict(result)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def write_full_yaml(
        result: CalibrationResult, path: str,
        board=None,
        corner_subsample: str = "all",
        config_path: str = "",
        detection_filter_stats: dict | None = None,
    ) -> None:
        """Write full replay YAML (File 2)."""
        data = OutputWriter.to_full_dict(
            result, board=board, corner_subsample=corner_subsample,
            config_path=config_path,
            detection_filter_stats=detection_filter_stats,
        )
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def write_yaml(result: CalibrationResult, path: str) -> None:
        """Write result to a YAML file (back-compat: calibration + timings)."""
        data = OutputWriter.to_dict(result)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def write_json(result: CalibrationResult, path: str) -> None:
        """Write result to a JSON file."""
        data = OutputWriter.to_dict(result)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=_ndarray_to_list)

    @staticmethod
    def write(result: CalibrationResult, path: str, fmt: str = "yaml") -> None:
        """Write result to file in specified format."""
        if fmt == "yaml":
            OutputWriter.write_yaml(result, path)
        elif fmt == "json":
            OutputWriter.write_json(result, path)
        else:
            raise ValueError(f"Unknown output format: {fmt}")
