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
"""High-level report generation function.

Single entry point for generating HTML calibration reports from
CalibrationResult + frame data. Used by all scripts.

The per-observation error computed here goes through `ResultProjector`: the forward model
the optimizer minimizes, against the board the solve fitted (refined when it estimated the
board shape) — so report residuals agree with the optimizer's cost by construction.
"""

import logging
from collections import defaultdict

import numpy as np

from mvis.detection.board import Board
from mvis.core.types import CalibrationResult, FrameObservations
from mvis.analysis.html_report import generate_html_report
from mvis.analysis.plot_3d import plot_camera_poses
from mvis.analysis.pose_images import camera_views
from mvis.calibration.result_projector import ResultProjector

logger = logging.getLogger(__name__)


def generate_calibration_report(
    result: CalibrationResult,
    all_frames: dict[int, list[FrameObservations]],
    board: Board,
    calibrations: dict[int, object],
    image_sizes: dict[int, tuple[int, int]],
    title: str = "Calibration Report",
    output_path: str = "report.html",
    sample_detection_images: dict[int, list[np.ndarray]] | None = None,
    all_detection_images: dict[int, list[np.ndarray]] | None = None,
    detection_stats: dict | None = None,
    calibration_info: dict | None = None,
    thumb_dir=None,
    appendix_max_frames: int = 0,
):
    """Generate a complete HTML calibration report.

    This is the main entry point for report generation. It computes error
    analysis, generates 3D visualization, and writes a self-contained HTML file.

    Args:
        result: CalibrationResult from optimizer.
        all_frames: camera_id -> list of FrameObservations.
        board: Board object.
        calibrations: camera_id -> GTSAM calibration object (Cal3DS2 or Cal3Fisheye).
            (Kept for backward compatibility; the report now rebuilds the
            calibration from `result.cameras` so the numbers match the factor.)
        image_sizes: camera_id -> (width, height).
        title: Report title.
        output_path: Path to write HTML file.
        sample_detection_images: camera_id -> list of 3-5 sample annotated images.
        all_detection_images: camera_id -> list of ALL annotated images for appendix.
        detection_stats: camera_id -> {total_frames, detected_frames, mean_tags, ...}.
        thumb_dir: run's `thumbnails/` folder; its per-camera index puts the real
            frame on the 3D plot's moving frustum (rendered view otherwise).
    """
    # The result's projection: its cameras, extrinsics (T_to_from = T_ref_cam, the factor's T_ext),
    # valid ref-camera pose timeline, and the board the solve fitted (refined when estimated).
    projector = ResultProjector(result, board, image_sizes)
    cam_params_map = {cam.camera_id: cam for cam in result.cameras}
    ref_cam = result.reference_camera

    active_cams = [cam.camera_id for cam in result.cameras]

    # Compute error data per camera using the SHARED projection model
    error_data = {}
    for cam_id in active_cams:
        frames = all_frames.get(cam_id, [])
        if not frames:
            continue
        if cam_id not in cam_params_map:
            continue

        cam = projector.camera(cam_id)
        # frames off every pose node were dropped by the solve; not scored (NaN). A non-ref camera
        # that would lose over half its frames is an unsynced rig the solve kept interpolating
        # (CalibrationPipeline._drop_unpaired_nonref leaves it alone), so it keeps them here too.
        brackets = [projector.solved_bracket(cam, f.timestamp) for f in frames]
        if not cam.is_ref and 2 * sum(b is None for b in brackets) > len(brackets):
            brackets = [projector.bracket(f.timestamp) for f in frames]

        frame_rms = []
        frame_median = []
        frame_p95 = []
        all_errors = []
        all_vecs = []
        error_vectors_by_frame = {}
        corner_errors = defaultdict(list)
        tag_errors = defaultdict(list)
        tag_det_counts = defaultdict(int)
        frame_tag_errors = {}

        # Corners per tag: 4 for AprilTag, 1 for chessboard
        cpt = 4 if hasattr(board, "_num_tags") and board._num_tags > 0 else 1

        for fi, frame in enumerate(frames):
            ferrs = []
            fvecs = []
            per_tag = defaultdict(list)
            det_tags = set()

            bracket = brackets[fi]
            if bracket is None:
                frame_rms.append(float("nan"))
                frame_median.append(float("nan"))
                frame_p95.append(float("nan"))
                continue

            for obs in frame.observations:
                det_tags.add(obs.point_id // cpt)
                proj = projector.project(cam, obs, bracket)
                if proj is None:
                    continue
                ex = float(proj[0] - obs.pixel[0])
                ey = float(proj[1] - obs.pixel[1])
                err = float(np.hypot(ex, ey))
                ferrs.append(err)
                fvecs.append([ex, ey])
                all_errors.append(err)
                all_vecs.append([ex, ey])
                corner_errors[obs.point_id].append(err)
                tid = obs.point_id // cpt
                tag_errors[tid].append(err)
                per_tag[tid].append(err)

            for tid in det_tags:
                tag_det_counts[tid] += 1

            if ferrs:
                arr = np.asarray(ferrs, dtype=np.float64)
                frame_rms.append(float(np.sqrt(np.mean(arr ** 2))))
                frame_median.append(float(np.median(arr)))
                frame_p95.append(float(np.percentile(arr, 95)))
                error_vectors_by_frame[fi] = np.asarray(fvecs, dtype=np.float64)
            else:
                frame_rms.append(float("nan"))
                frame_median.append(float("nan"))
                frame_p95.append(float("nan"))

            frame_tag_errors[fi] = {
                t: float(np.sqrt(np.mean(np.array(e) ** 2)))
                for t, e in per_tag.items()
            }

        error_data[cam_id] = {
            "frame_rms": frame_rms,
            "frame_median": frame_median,
            "frame_p95": frame_p95,
            "errors": np.array(all_errors),
            "error_vectors": np.array(all_vecs) if all_vecs else np.zeros((0, 2)),
            "error_vectors_by_frame": error_vectors_by_frame,
            "corner_errors": dict(corner_errors),
            "tag_errors": dict(tag_errors),
            "frame_tag_errors": frame_tag_errors,
            "tag_detection_counts": dict(tag_det_counts),
        }

    # 3D visualization. Skip if board_poses are missing (e.g. when
    # regenerating HTML from disk where the YAML never serialized them).
    keep = [i for i, p in enumerate(result.board_poses)
            if p is not None and not np.allclose(p, np.eye(4))]
    valid_poses = [result.board_poses[i] for i in keep]
    times = result.board_pose_times
    valid_times = ([times[i] for i in keep]
                   if len(times) == len(result.board_poses) else None)
    if valid_poses:
        ext_for_plot = [(ext.from_camera, ext.T_to_from)
                        for ext in result.extrinsics] or None
        views = None
        if valid_times is not None and ref_cam in cam_params_map:
            try:
                views = camera_views(
                    valid_poses, valid_times, result.cameras,
                    {ext.from_camera: ext.T_to_from for ext in result.extrinsics},
                    ref_cam, board, all_frames, thumb_dir=thumb_dir,
                    time_offsets=getattr(result, "time_offsets", None))
            except Exception as e:      # images are garnish; never fail the report
                logger.warning(f"frustum images / side panel skipped: {e}")
        fig_3d = None
        try:   # the canvas views (analysis.views); the Plotly figure stays as the fallback
            from mvis.analysis.views.spec import cam_only_rig_spec, cam_only_scene_spec
            ext_by_id = {ext.from_camera: ext.T_to_from for ext in result.extrinsics}
            view_specs = {"scene": cam_only_scene_spec(valid_poses, valid_times, board, result.cameras, ext_by_id, ref_cam, views),
                          "rig": cam_only_rig_spec(result.cameras, ext_by_id, ref_cam,
                                                  {e.camera_id: (e.median, e.p95)
                                                   for e in (result.reprojection_errors or [])})}
        except Exception as e:
            logger.warning(f"report views failed; using the Plotly figures: {e}")
            view_specs = {}
        if not view_specs.get("scene"):
            fig_3d = plot_camera_poses(valid_poses, board,
                                        extrinsics=ext_for_plot, title=title,
                                        times=valid_times, views=views,
                                        corner_subsample=(calibration_info or {}).get(
                                            "corner_subsample", "all"))
    else:
        fig_3d, view_specs = None, {}

    # Auto-compute per-camera effective frequency for config section
    auto_info = {"cameras_info": {}}
    for cam_id, frames in all_frames.items():
        if len(frames) < 2:
            continue
        ts = sorted(f.timestamp for f in frames)
        dur = ts[-1] - ts[0]
        hz = (len(ts) - 1) / dur if dur > 0 else 0.0
        auto_info["cameras_info"][cam_id] = {
            "n_frames_used": len(frames),
            "effective_hz": float(hz),
            "duration_s": float(dur),
        }
    if calibration_info:
        # Merge user-provided info over auto-computed info
        merged = dict(auto_info)
        merged.update({k: v for k, v in calibration_info.items()
                       if k != "cameras_info"})
        ci_user = calibration_info.get("cameras_info", {})
        for cid, d in auto_info["cameras_info"].items():
            d.update(ci_user.get(cid, {}))
        merged["cameras_info"] = auto_info["cameras_info"]
        # Carry over any user-supplied cams not in auto_info
        for cid, d in ci_user.items():
            merged["cameras_info"].setdefault(cid, d)
        calibration_info = merged
    else:
        calibration_info = auto_info

    # Compute motion excitation metrics from board poses
    from mvis.analysis.motion_analysis import (
        compute_motion_metrics, hull_coverage, motion_series, valid_board_poses,
    )
    motion_metrics = {}
    motion_omega_data = {}  # for angular velocity plot

    if result.board_poses and result.board_pose_times:
        # Compute for the reference camera (board poses are ref-cam trajectory)
        m = compute_motion_metrics(
            result.board_poses, result.board_pose_times, ref_cam)

        # Hull coverage for distortion assessment
        ref_img_sz = image_sizes.get(ref_cam, (640, 480))
        m.hull_coverage_pct = hull_coverage(
            [obs.pixel for frame in all_frames.get(ref_cam, []) for obs in frame.observations],
            ref_img_sz[0], ref_img_sz[1])[0]

        motion_metrics[ref_cam] = m

        # Angular velocity timeline (||omega|| frame-invariant; camera-frame x, y)
        series = motion_series(valid_board_poses(result.board_poses, result.board_pose_times))
        omega_ts = [t for t, _dt, _r, _w in series]
        omega_norms = [float(np.linalg.norm(r / dt)) for _t, dt, r, _w in series]
        omega_xs_plot = [float(w[0]) for *_x, w in series]
        omega_ys_plot = [float(w[1]) for *_x, w in series]
        omega_perps_plot = [float(np.sqrt(w[0]**2 + w[1]**2)) for *_x, w in series]

        motion_omega_data[ref_cam] = {
            "timestamps": omega_ts,
            "omega_norms": omega_norms,
            "omega_perps": omega_perps_plot,
            "omega_xs": omega_xs_plot,
            "omega_ys": omega_ys_plot,
        }

    # Attach motion metrics to result for YAML serialization
    if motion_metrics:
        result.motion_excitation = {
            cam_id: m.to_dict() for cam_id, m in motion_metrics.items()
        }

    # Generate HTML
    generate_html_report(
        result=result,
        all_frames=all_frames,
        board=board,
        detection_images=sample_detection_images or {},
        error_data=error_data,
        plot_3d_fig=fig_3d,
        views=view_specs,
        title=title,
        output_path=output_path,
        all_detection_images=all_detection_images,
        detection_stats=detection_stats,
        calibration_info=calibration_info,
        motion_metrics=motion_metrics,
        motion_omega_data=motion_omega_data,
        appendix_max_frames=appendix_max_frames,
    )

    logger.info(f"Report generated: {output_path}")
