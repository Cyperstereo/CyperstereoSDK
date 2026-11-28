# OpenMVIS: A Multi-Visual-Inertial Sensor Calibration Benchmark
# Copyright (C) 2026 Yulin Yang
# Copyright (C) 2026 Yuantai-Z
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
# SPDX-FileCopyrightText: 2026 Yuantai-Z
# SPDX-License-Identifier: GPL-3.0-only
"""Multi-camera calibration pipeline.

Implements a 5-stage pipeline:
  1. Per-camera initialization (OpenCV intrinsics + PnP board poses)
  2. Per-camera GTSAM optimization (refine intrinsics + poses per camera)
  3. Rolling shutter detection (row-correlated residual analysis)
  4. Trajectory alignment (extrinsics + time offsets via RANSAC)
  5. Joint multi-camera optimization (all parameters in one factor graph)

Usage:
    pipeline = CalibrationPipeline(board, config)
    result = pipeline.run(all_frames, image_sizes)
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import gtsam

from mvis.detection.board import Board
from mvis.calibration.initializer import Initializer
from mvis.calibration.optimizer import Optimizer, OptimizerOptions
from mvis.calibration.omni_schedule import (
    PHASED_MODELS, poses_by_frame, solve_phased)
from mvis.calibration.pose_alignment import _BOARD_Z_RANGE
from mvis.io.frame_selector import nearest_ref
from mvis.calibration.lm_params import LM_LAMBDA_LOWER_DEFAULT, LM_LAMBDA_UPPER_DEFAULT
from mvis.core import shutter
from mvis.core.camera_models import (UCM_ALPHA_RANGE, canonical_dist, default_dist,
                                     KB_THETA_MODELS, dist_sanity_issue, n_dist)
from mvis.core.timing import Timer
from mvis.core.types import CalibrationResult, FrameObservations

logger = logging.getLogger(__name__)


# Thresholds for the pass-2 sanity guardrail. Conservative — only trip on
# obvious pathology, not a mild regression (pass-2 is often 0.01-0.05 px
# worse on P95 just from trimming tail, which is fine).
_SANITY_FX_MIN = 1.0                   # px; fx must be positive
_SANITY_FX_FY_MAX_ASYM = 0.30          # |fx - fy| / max(fx, fy)
_SANITY_DISTORTION_MAX_ABS = 2.0       # any |dist_coeff| above this is bogus
                                       # (omni xi: [0, 5]; ucm alpha: [0, 1); see
                                       # camera_models.dist_sanity_issue)
_SANITY_P95_REGRESSION = 1.5           # pass-2 P95 > 1.5× pass-1 P95 on any cam


def _detect_diverged_poses_from_state(
    poses: list,
    min_z: float = _BOARD_Z_RANGE[0],
    max_z: float = _BOARD_Z_RANGE[1],
) -> set[int]:
    """Detect diverged board poses from state.board_poses[cid].

    A pose is diverged if the board's Z-depth in camera frame is outside
    [min_z, max_z]. This catches:
    - Identity poses (z=0, from optimizer sanity replacement)
    - Poses pushed behind camera (z<0)
    - Poses pushed to infinity (z>20m)

    Args:
        poses: List of Pose3 or None (frame-indexed).
        min_z: Minimum board distance in camera frame (meters).
        max_z: Maximum board distance in camera frame (meters).

    Returns:
        Set of frame indices with diverged poses.
    """
    diverged = set()
    for fi, pose in enumerate(poses):
        if pose is None:
            continue
        T_cam_board = pose.inverse()
        z = T_cam_board.translation()[2]
        if z < min_z or z > max_z:
            diverged.add(fi)
    return diverged


_STAGE1_DIST_MAX_ABS = 5.0             # Stage-1 init: any |dist_coeff| above this is bogus


def _stage1_dist_extreme(dm: str, dist) -> bool:
    """Stage-1 init sanity on the distortion: any |c| > _STAGE1_DIST_MAX_ABS.
    ucm: alpha outside UCM_ALPHA_RANGE [0, 1) as well."""
    d = np.asarray(dist, float)
    if dm == "ucm" and not UCM_ALPHA_RANGE[0] <= d[0] < UCM_ALPHA_RANGE[1]:
        return True
    return bool(np.any(np.abs(d) > _STAGE1_DIST_MAX_ABS))


def _fallback_dist(dm: str) -> np.ndarray:
    """Distortion of a seed without one, or of the Stage-1 heuristic fallback:
    zeros (padded to >= 4, the K layout), except ucm, whose zero alpha is a
    pinhole that drops every ray past 90 deg — it takes its registry default."""
    d0 = default_dist(dm) if dm == "ucm" else [0.0] * n_dist(dm)
    return np.array(d0 + [0.0] * max(0, 4 - len(d0)))


def _log_kb_valid_domain(result, image_sizes) -> None:
    """kb4 cameras of a finished solve: log the fit's valid domain (first
    theta_d fold vs the image corner); a WARNING, never a guardrail failure,
    when it folds inside the image. Observed corners past the fold are the
    alarm level: louder, and the solve still returns. Equidistant cameras: a WARNING naming kb4 when
    corners fell out past 90 deg."""
    if result is None:
        return
    from mvis.calibration.corner_accounting import warn_past_90
    accounting = getattr(result, "corner_accounting", None) or {}
    for cam in result.cameras:
        if cam.distortion_model in KB_THETA_MODELS:
            from mvis.camera.kb_domain import check_kb_valid_domain
            size = (cam.image_size if min(cam.image_size) > 0
                    else image_sizes.get(cam.camera_id))
            check_kb_valid_domain(np.r_[cam.intrinsics, cam.distortion], size,
                                  cam.distortion_model,
                                  label=f"Camera {cam.camera_id}", log=logger)
            acc = accounting.get(cam.camera_id) or {}
            if acc.get("fold_level") == "alarm":
                logger.warning(f"Camera {cam.camera_id}: ALARM: {acc['past_fold']} observed "
                               f"corners lie past the {cam.distortion_model} fold at the final "
                               f"fit and are not in its error ({acc['past_fold_pass1']} at the "
                               f"first pass)")
        warn_past_90(accounting.get(cam.camera_id), f"Camera {cam.camera_id}", logger)


def _diverged_pose_flags(result) -> list:
    """Per board pose, whether final cleanup discards it: identity, board behind or too far, an
    isolated translation jump, or a non-ref camera behind the board."""
    n_bp = len(result.board_poses)
    flagged = [False] * n_bp

    # Pass 1: identity + z-depth
    # Check both directions:
    #   - Board z in camera frame (T_inv[2,3]): must be > 0.05
    #   - Camera z in board frame (T[2,3]): must be > 0 (camera in front of board)
    translations = []
    for i, bp in enumerate(result.board_poses):
        if bp is None:
            translations.append(None)
            continue
        if np.allclose(bp, np.eye(4)):
            flagged[i] = True
            translations.append(None)
            continue
        T_inv = np.linalg.inv(bp)
        z_board_in_cam = T_inv[2, 3]
        z_cam_in_board = bp[2, 3]
        t = T_inv[:3, 3]
        translations.append(t)
        if z_board_in_cam < _BOARD_Z_RANGE[0] or z_board_in_cam > _BOARD_Z_RANGE[1]:
            flagged[i] = True
        elif z_cam_in_board < 0:
            flagged[i] = True  # camera behind board plane

    # Pass 2: translation-jump outlier detection.
    # A pose is flagged if its distance to BOTH temporal neighbors
    # exceeds jump_factor * median neighbor distance.
    valid_trans = [(i, t) for i, t in enumerate(translations)
                   if t is not None and not flagged[i]]
    if len(valid_trans) >= 3:
        neighbor_dists = []
        for k in range(1, len(valid_trans)):
            d = np.linalg.norm(valid_trans[k][1] - valid_trans[k-1][1])
            neighbor_dists.append(d)
        med_nd = float(np.median(neighbor_dists))
        jump_threshold = max(5.0 * med_nd, 0.5)  # at least 0.5m

        for k in range(len(valid_trans)):
            idx = valid_trans[k][0]
            t_cur = valid_trans[k][1]
            d_prev = (np.linalg.norm(t_cur - valid_trans[k-1][1])
                      if k > 0 else 0.0)
            d_next = (np.linalg.norm(t_cur - valid_trans[k+1][1])
                      if k < len(valid_trans) - 1 else 0.0)
            # Flag if jumps in BOTH directions (isolated spike)
            if d_prev > jump_threshold and d_next > jump_threshold:
                flagged[idx] = True

    # Pass 3: check non-ref camera positions via extrinsics.
    # After composing T_board_ref @ T_ref_cam, the non-ref camera
    # may end up behind the board (z < 0 in board frame).
    if result.extrinsics:
        for ext in result.extrinsics:
            T_ext = ext.T_to_from  # T_ref_cam
            for i, bp in enumerate(result.board_poses):
                if bp is None or flagged[i]:
                    continue
                T_board_cam = bp @ T_ext
                cam_z = T_board_cam[2, 3]
                if cam_z < 0:
                    flagged[i] = True
    return flagged


def _pass2_is_pathological(pass1: CalibrationResult,
                           pass2: CalibrationResult) -> bool:
    """Return True if pass-2 should be rejected in favour of pass-1.

    Checks for: (1) non-positive focal lengths, (2) extreme fx/fy
    asymmetry, (3) blown-out distortion coefficients, (4) per-camera P95
    regression > 1.5×. Logs the specific reason for downstream debugging.
    """
    # (1)-(3): intrinsic sanity
    for cam in pass2.cameras:
        fx = float(cam.intrinsics[0])
        fy = float(cam.intrinsics[1])
        if fx <= _SANITY_FX_MIN or fy <= _SANITY_FX_MIN:
            logger.warning(
                f"  cam{cam.camera_id}: fx={fx:.2f} fy={fy:.2f} (non-positive)"
            )
            return True
        asym = abs(fx - fy) / max(fx, fy)
        if asym > _SANITY_FX_FY_MAX_ASYM:
            logger.warning(
                f"  cam{cam.camera_id}: fx/fy asymmetry {asym*100:.1f}% > "
                f"{_SANITY_FX_FY_MAX_ASYM*100:.0f}%"
            )
            return True
        # per-model rule shared with the regression health check
        issue = dist_sanity_issue(cam.distortion_model, cam.distortion,
                                  _SANITY_DISTORTION_MAX_ABS)
        if issue is not None:
            i, reason = issue
            if reason is None:
                c = cam.distortion[i]
                reason = (f"distortion[{i}]={float(c):.3f} "
                          f"exceeds |{_SANITY_DISTORTION_MAX_ABS}|")
            logger.warning(f"  cam{cam.camera_id}: {reason}")
            return True

    # (4): per-camera P95 regression — but ONLY when the camera's
    # intrinsics are stable (∆fx < 5 %). When pass-2 intentionally moves
    # to a different optimization basin (e.g. virig_flir_t265 bag1 where
    # cam0 fx dropped from 508→453 in Stage 5, matching the standalone
    # single-cam result), the P95 comparison is between two DIFFERENT
    # calibrations on different filtered observation sets — not directly
    # meaningful. In that case we trust the overall RMS signal below.
    p1_cams = {c.camera_id: c for c in pass1.cameras}
    p1_by_cam = {e.camera_id: e for e in pass1.reprojection_errors}
    for e2 in pass2.reprojection_errors:
        e1 = p1_by_cam.get(e2.camera_id)
        if e1 is None or e1.p95 <= 0:
            continue
        ratio = e2.p95 / e1.p95
        if ratio > _SANITY_P95_REGRESSION:
            # Check whether intrinsics changed significantly
            c2 = next((c for c in pass2.cameras
                       if c.camera_id == e2.camera_id), None)
            c1 = p1_cams.get(e2.camera_id)
            fx_changed = False
            if c1 is not None and c2 is not None:
                fx1, fx2 = float(c1.intrinsics[0]), float(c2.intrinsics[0])
                if fx1 > 0:
                    fx_changed = abs(fx2 - fx1) / fx1 > 0.05
            if fx_changed:
                logger.info(
                    f"  cam{e2.camera_id}: P95 regressed {e1.p95:.3f}→{e2.p95:.3f} "
                    f"px ({ratio:.2f}×) BUT fx moved {fx1:.1f}→{fx2:.1f} — pass-2 "
                    "is in a different basin, deferring to overall-RMS check below"
                )
                continue
            logger.warning(
                f"  cam{e2.camera_id}: P95 regressed {e1.p95:.3f}→{e2.p95:.3f} "
                f"px ({ratio:.2f}× > {_SANITY_P95_REGRESSION}×, fx stable)"
            )
            return True

    # (5): overall RMS regression across all cameras. When a per-camera
    # P95 check was deferred above (because intrinsics moved), this is
    # the authoritative check. Pass-2 should improve the overall fit;
    # if it doesn't (mean RMS regressed >1.5×), revert.
    p1_rms = [e.rms for e in pass1.reprojection_errors if e.rms > 0]
    p2_rms = [e.rms for e in pass2.reprojection_errors if e.rms > 0]
    if p1_rms and p2_rms:
        m1 = sum(p1_rms) / len(p1_rms)
        m2 = sum(p2_rms) / len(p2_rms)
        if m2 > m1 * _SANITY_P95_REGRESSION:
            logger.warning(
                f"  overall mean RMS regressed {m1:.3f}→{m2:.3f} px "
                f"({m2/m1:.2f}× > {_SANITY_P95_REGRESSION}×)"
            )
            return True
    return False


@dataclass
class PipelineConfig:
    """Configuration for the calibration pipeline."""

    # Per-camera init
    max_init_frames: int = 100  # max frames for OpenCV calibrateCamera init (0 = no cap)
    init_frame_selection: str = "uniform"  # "uniform" (time-spaced) or "diverse" (position/size/orientation bins)

    # Optimizer
    max_iterations: int = 15
    max_iterations_single: int = 15
    max_iterations_rerun: int = 5  # warm-started rerun converges fast
    pixel_sigma: float = 1.0
    huber_threshold: float = 1.345
    # LM damping bounds (cam-imu-speedup L1; see `mvis.calibration.lm_params`).
    # NOTE: the LM termination tolerances on this path are OptimizerOptions'
    # own (1e-8 abs / 1e-5 rel), not config.optimizer.*_error_tol.
    lm_lambda_lower: float = LM_LAMBDA_LOWER_DEFAULT
    lm_lambda_upper: float = LM_LAMBDA_UPPER_DEFAULT

    # Stage 2 input pre-filter: score each candidate frame against the
    # Stage 1 OpenCV calibration (radtan only — other models use VP init
    # with hardcoded distortion, no trusted reference). Drop frames with
    # median reprojection residual > max(2 px, p99 of per-frame medians).
    # Prevents Stage 2 from being pulled into a wrong basin by redundant
    # or low-quality frames the diversity filter retained for time-offset
    # cross-correlation. Per-camera independent. Skipped if the filter
    # would drop > 30 % of frames (sanity gate). Found 2026-05-02 via
    # virig_flir_t265 bag1 wrong-basin convergence on cam0 (Blackfly).
    stage2_pre_filter: bool = True
    stage2_pre_filter_min_threshold_px: float = 2.0
    stage2_pre_filter_percentile: float = 99.0
    stage2_pre_filter_max_drop_fraction: float = 0.60

    # Feature flags
    optimize_intrinsics: bool = True
    optimize_time_offset: bool = True

    # Reference camera (None = auto-select camera with most frames)
    reference_camera: int | None = None

    # Camera distortion models: cam_id -> "radtan" or "equidistant"
    distortion_models: dict[int, str] = field(default_factory=dict)

    # Per-tag corner subsampling (AprilGrid only). "all" keeps every
    # observation; "top_left"/"tl", "bl", "br", "tr", "two_diag",
    # "two_anti" thin the observations fed into Stages 2-4. Stage 1
    # (OpenCV intrinsic init + PnP) always uses every corner so the
    # initial guess isn't degraded.
    corner_subsample: str = "two_anti"

    # Detection-quality gating (between Stage 1 and Stage 2).
    # Pooled-pose filter: a ref-camera frame survives iff total tags
    # observed across all cameras within the sync window
    # is >= max(min_pose_tag_floor, ceil(min_pose_tag_fraction × num_expected)).
    enable_pose_filter: bool = True
    min_pose_tag_fraction: float = 0.20
    min_pose_tag_floor: int = 4
    pose_sync_tolerance_s: float | None = None  # None = auto from ref dt
    # Joint-solve pairing (no time offset estimated): a non-ref frame is kept
    # only if a ref frame (= pose node) exists within this many seconds;
    # otherwise it would be interpolated between ref poses far apart. Kept
    # frames use that ref pose exactly (OptimizerOptions.snap_sync_tolerance_s).
    # Below half a frame period for <= 60 Hz cameras; above the T265 ~2.5 ms
    # fisheye1/fisheye2 stamp offset.
    pair_sync_tolerance_s: float = 0.008
    # Per-corner observability filter: drop corners with < min_corner_views
    # total observations across all (camera, frame) pairs.
    enable_corner_observability_filter: bool = True
    min_corner_views: int = 5
    # Per-camera corner filter: drop corners seen < N times in each camera
    min_corner_views_per_camera: int = 3
    # Per-frame dual filter (both must pass):
    min_tags_per_frame: int = 6   # LEGACY — kept for backward compat
    min_tag_count: int = 6        # min unique tags (spatial spread)
    min_point_count: int = 0      # min observation points (0 = tag-only filter)
    adaptive_subsample: bool = True  # upgrade high-quality frames with more corners (Stage 4)
    # PnP: discard poses with median reprojection error above this
    max_pnp_reprojection_px: float = 10.0
    pnp_method: str = "iterative"   # 'ippe' = planar-safe board PnP (MOC fix)
    # Board pose + per-corner heights; see OptimizerOptions.
    board_heights: bool = False
    board_height_sigma: float = 0.003
    # Validate Stage 1 intrinsics (fx>0, symmetric, sane cx/cy)
    stage1_intrinsic_validation: bool = True
    # Tag spread: minimum convex hull of tag centers / image area
    min_hull_fraction: float = 0.01

    # Per-camera configs (CameraConfig list) — used to read shutter_type
    # per camera for RS calibration. Optional; if empty, all cameras
    # default to global shutter (no RS).
    camera_configs: list = field(default_factory=list)

    # Outlier-driven rerun: after Stage 4, drop both (a) frames with high
    # median residual and (b) individual observations with high residual,
    # then rerun Stage 4 warm-started from the first-pass values.
    rerun_after_outlier_rejection: bool = False
    outlier_threshold_px: float = 2.0            # per-obs absolute floor
    outlier_percentile: float = 99.0             # per-obs percentile
    frame_median_threshold_px: float | None = None  # per-frame median floor
    frame_median_percentile: float = 99.0        # per-frame median percentile

    # Verbose output
    verbose: bool = True

    # Diagnostic: freeze intrinsics for these cameras during joint optimization
    frozen_intrinsics_cameras: set = None
    # Diagnostic: freeze RS readout at Stage 2 value for these cameras
    freeze_rs_cameras: set = None

    # Per-camera Stage 2 target rate. If None, Stage 2 uses the input
    # frame rate (no additional subsampling). Float for uniform rate,
    # or dict {cam_id: hz} for per-camera rates.
    # Recommended: GS=5Hz, RS=10Hz — intrinsics don't need high frame rate.
    stage2_target_hz: float | dict[int, float] | None = None

    # Rerun visualization: launch Rerun viewer after calibration to display
    # board poses, camera trajectory, extrinsics, and detection images.
    rerun_visualization: bool = False

    @classmethod
    def from_calibration_config(cls, config, *, reference_camera=None,
                                optimize_time_offset=None,
                                rerun_after_outlier_rejection=None,
                                corner_subsample=None, distortion_models=None,
                                frozen_intrinsics_cameras=None,
                                freeze_rs_cameras=None, stage2_target_hz=None,
                                rerun_visualization=False, verbose=True):
        """Build a PipelineConfig from a CalibrationConfig — the single converter.

        Optimizer numerics come from ``config.optimizer.*`` (NO hardcodes; the
        exception is the LM termination tolerance pair, which this path leaves at
        ``OptimizerOptions``' own defaults, see ``PipelineConfig.lm_lambda_*``); runtime/
        derived fields (reference camera, env-driven freezes, the computed Stage-2
        rate, rerun viz) are passed as keyword overrides. An override that is not
        None wins over the config value (e.g. a CLI ``--corner-subsample``).
        ``config`` is duck-typed (a ``core.config.CalibrationConfig``) to avoid an
        import cycle.
        """
        opt = config.optimizer
        dist = (distortion_models if distortion_models is not None
                else {c.camera_id: c.distortion_model for c in config.cameras})
        ot = (optimize_time_offset if optimize_time_offset is not None
              else config.optimize_time_offset)
        rr = (rerun_after_outlier_rejection if rerun_after_outlier_rejection
              is not None else config.rerun_outliers)
        cs = corner_subsample if corner_subsample is not None else config.corner_subsample
        ref = reference_camera if reference_camera is not None else config.reference_camera
        return cls(
            max_init_frames=opt.max_init_frames,
            init_frame_selection=opt.init_frame_selection,
            pnp_method=getattr(opt, "pnp_method", "iterative"),
            max_iterations=opt.max_iterations,
            max_iterations_single=opt.max_iterations,
            pixel_sigma=opt.pixel_sigma,
            huber_threshold=opt.huber_threshold,
            lm_lambda_lower=getattr(opt, "lm_lambda_lower", LM_LAMBDA_LOWER_DEFAULT),
            lm_lambda_upper=getattr(opt, "lm_lambda_upper", LM_LAMBDA_UPPER_DEFAULT),
            optimize_intrinsics=opt.optimize_intrinsics,
            optimize_time_offset=ot,
            reference_camera=ref,
            distortion_models=dist,
            corner_subsample=cs,
            rerun_after_outlier_rejection=rr,
            camera_configs=list(config.cameras),
            frozen_intrinsics_cameras=frozen_intrinsics_cameras,
            freeze_rs_cameras=freeze_rs_cameras,
            stage2_target_hz=stage2_target_hz,
            rerun_visualization=rerun_visualization,
            # estimate.board_points -> per-corner board heights (was only wired on
            # the `mvis cam` CLI path; scripts/run_calibration.py silently solved a
            # flat board until 2026-11-01)
            board_heights=bool(getattr(opt, "optimize_board_points", False)),
            verbose=verbose,
        )


@dataclass
class PipelineState:
    """Intermediate state passed between pipeline stages."""

    # Per-camera data from stage 1 (init) and stage 2 (single-cam optimize)
    intrinsics: dict[int, np.ndarray] = field(default_factory=dict)
    dist_coeffs: dict[int, np.ndarray] = field(default_factory=dict)
    board_poses: dict[int, list] = field(default_factory=dict)  # cam_id -> list[Pose3|None]
    # 2c-8b: poses keyed by frame IDENTITY (timestamp), not list position —
    # the index heuristic this replaces is documented at the prefilter
    pose_by_ts: dict[int, dict] = field(default_factory=dict)
    distortion_models: dict[int, str] = field(default_factory=dict)
    single_cam_results: dict[int, CalibrationResult] = field(default_factory=dict)

    # RS cameras from config
    rs_cameras: dict[int, float] = field(default_factory=dict)  # cam_id -> prior_sigma
    rs_readouts: dict[int, float] = field(default_factory=dict)  # cam_id -> readout (seconds) from Stage 2

    # From Stage 3 (trajectory alignment)
    extrinsics: dict[int, gtsam.Pose3] = field(default_factory=dict)
    time_offsets: dict[int, float] = field(default_factory=dict)
    outlier_frame_timestamps: dict[int, set[float]] = field(default_factory=dict)

    # Timing tree (hierarchical wall-clock). Pipeline.run() initialises
    # this. Use ``state.timer.section("name")`` as a context manager.
    timer: object = None  # mvis.core.timing.Timer instance


def _warm_start_extrinsics(result) -> dict:
    """Pass-1 extrinsics in the convention Stage 4 consumes (state.extrinsics
    holds T_cam_ref, as Stage 3 produces; Optimizer.solve inserts
    ``ext.inverse()`` as its E = T_ref_cam variable). A result's
    ``T_to_from`` IS that E (T_ref_cam), so it must be inverted here."""
    return {ext.from_camera: gtsam.Pose3(ext.T_to_from).inverse()
            for ext in result.extrinsics}


def _stage_banner(msg: str) -> None:
    """Log a visual stage separator."""
    logger.info("=" * 60)
    logger.info(msg)
    logger.info("=" * 60)


def _log_stage_comparison(label: str,
                          prev_results: dict,
                          curr_result,
                          ) -> None:
    """Log intrinsic + error comparison between two stages.

    prev_results: dict[cam_id → CalibrationResult] from Stage 2 single-cam,
                  or a CalibrationResult from a previous joint stage.
    curr_result:  CalibrationResult from the current stage.
    """
    _stage_banner(label)

    # Build lookup for previous per-camera params and errors
    prev_cams = {}
    prev_errs = {}
    if isinstance(prev_results, dict):
        # Stage 2: dict of cam_id → CalibrationResult
        for cid, res in prev_results.items():
            if res and res.cameras:
                prev_cams[cid] = res.cameras[0]
            if res and res.reprojection_errors:
                prev_errs[cid] = res.reprojection_errors[0]
    else:
        # CalibrationResult from a previous joint stage
        for cam in prev_results.cameras:
            prev_cams[cam.camera_id] = cam
        for err in prev_results.reprojection_errors:
            prev_errs[err.camera_id] = err

    def _pct(a, b):
        if b and abs(b) > 1e-12:
            return (a - b) / abs(b) * 100
        return 0.0

    for cam in curr_result.cameras:
        cid = cam.camera_id
        pc = prev_cams.get(cid)
        pe = prev_errs.get(cid)
        ce = None
        for e in curr_result.reprojection_errors:
            if e.camera_id == cid:
                ce = e
                break

        logger.info(f"  cam{cid}:")
        if pc is not None:
            for i, name in enumerate(["fx", "fy", "cx", "cy"]):
                old_v = pc.intrinsics[i]
                new_v = cam.intrinsics[i]
                d = _pct(new_v, old_v)
                flag = " !" if abs(d) > 1.0 else ""
                logger.info(f"    {name:>2}: {old_v:>9.2f} -> {new_v:>9.2f}  "
                            f"(d={d:+.2f}%){flag}")
            # Distortion
            old_d = [f"{v:.4f}" for v in pc.distortion]
            new_d = [f"{v:.4f}" for v in cam.distortion]
            logger.info(f"    dist: [{', '.join(old_d)}] -> [{', '.join(new_d)}]")
            # RS
            if cam.rolling_shutter_readout or pc.rolling_shutter_readout:
                old_rs = pc.rolling_shutter_readout * 1000
                new_rs = cam.rolling_shutter_readout * 1000
                d = _pct(new_rs, old_rs) if old_rs else 0
                logger.info(f"    RS: {old_rs:.2f}ms -> {new_rs:.2f}ms  (d={d:+.1f}%)")

        if pe is not None and ce is not None:
            d_med = _pct(ce.median, pe.median)
            flag = " !" if ce.median > 5 * pe.median else ""
            logger.info(f"    median: {pe.median:.2f}px -> {ce.median:.2f}px  "
                        f"(d={d_med:+.1f}%){flag}")
            logger.info(f"    P95:    {pe.p95:.2f}px -> {ce.p95:.2f}px")
            logger.info(f"    n_obs:  {pe.num_observations} -> {ce.num_observations}")


def _poses_by_identity(frames, full_frames, full_poses, pose_by_ts):
    """Board poses for `frames` by frame identity (timestamp).

    `full_poses` is positional to `full_frames`; `frames` may be a subset or
    a reordering of it. Falls back to `pose_by_ts` for frames not in
    `full_frames`, and to None when a frame has no pose anywhere.
    """
    if full_poses is not None and frames is full_frames:
        return list(full_poses)
    by_ts = dict(pose_by_ts or {})
    if full_poses is not None and len(full_poses) == len(full_frames):
        by_ts.update({f.timestamp: p for f, p in zip(full_frames, full_poses)})
    return [by_ts.get(f.timestamp) for f in frames]


class CalibrationPipeline:
    """Multi-camera calibration pipeline."""

    def __init__(self, board: Board, config: PipelineConfig | None = None,
                 ledger=None):
        self._board = board
        self._config = config or PipelineConfig()
        from mvis.calibration.selection_ledger import SelectionLedger
        self.ledger = ledger if ledger is not None else SelectionLedger()

    def _sync_pose_identity(self, all_frames, state, where: str) -> None:
        """2c-9 shadow half: after a stage that rewrites state.board_poses,
        assert the positional lockstep actually held (len(poses) ==
        len(frames) per camera) and refresh pose_by_ts from the zip, so the
        identity view is continuously current for later consumers. A length
        mismatch means the hand-maintained index dance broke — fail loudly."""
        for cid, poses in state.board_poses.items():
            frames = all_frames.get(cid)
            if frames is None:
                continue
            assert len(poses) == len(frames), (
                f"2c-9 lockstep broken {where} cam{cid}: "
                f"{len(poses)} poses vs {len(frames)} frames")
            state.pose_by_ts[cid] = {f.timestamp: p
                                     for f, p in zip(frames, poses)}

    def _drop_unpaired_nonref(self, frames, poses, ref_cam, stage):
        """Drop non-ref frames that have no ref frame at the same instant.

        Only when no time offset is estimated (the cameras are treated as
        synchronized): the joint solve has pose nodes on the ref camera only
        and would interpolate such a frame between ref poses >= 1/target_hz
        apart. Per-camera gates (min-tag, PnP, outlier rejection) can drop a
        ref frame while keeping its co-frames, which strands them.

        A camera that would lose more than half its frames (an
        unsynchronized rig) is left alone. Returns ``(frames, poses)``; both
        unchanged objects when nothing is dropped.
        """
        if (self._config.optimize_time_offset or ref_cam not in frames
                or len(frames) < 2):
            return frames, poses
        tol = self._config.pair_sync_tolerance_s
        # a ref frame is a pose node only if it has a pose and enough points
        # (Optimizer.solve skips the rest)
        min_pts = (self._board.min_points_for_pose()
                   if self._board is not None else 0)
        ref_plist = (poses or {}).get(ref_cam)
        ref_ts = np.sort([
            f.timestamp for i, f in enumerate(frames[ref_cam])
            if f.num_points >= min_pts and (
                ref_plist is None or i >= len(ref_plist)
                or ref_plist[i] is not None)])
        new_frames, new_poses, changed = {}, {}, False
        for cid, fr in frames.items():
            plist = poses.get(cid) if poses is not None else None
            if cid == ref_cam or len(ref_ts) == 0:
                new_frames[cid] = fr
                if plist is not None:
                    new_poses[cid] = plist
                continue
            ts = np.array([f.timestamp for f in fr])
            keep = nearest_ref(ts, ref_ts)[1] <= tol
            n_drop = int((~keep).sum())
            if n_drop == 0 or keep.sum() < 0.5 * len(fr):
                if n_drop:
                    logger.warning(
                        f"  cam{cid}: {n_drop}/{len(fr)} frames have no ref "
                        f"frame within {tol*1e3:.0f} ms — rig not synchronized? "
                        f"keeping them")
                new_frames[cid] = fr
                if plist is not None:
                    new_poses[cid] = plist
                continue
            changed = True
            new_frames[cid] = [f for f, k in zip(fr, keep) if k]
            if plist is not None:
                new_poses[cid] = [p for p, k in zip(plist, keep) if k]
            logger.info(f"  cam{cid}: dropped {n_drop} frames with no ref "
                        f"frame within {tol*1e3:.0f} ms ({stage})")
        if not changed:
            return frames, poses
        self.ledger.diff(frames, new_frames, stage, "unpaired_ref")
        if poses is not None:
            for cid, plist in poses.items():
                new_poses.setdefault(cid, plist)
        return new_frames, (new_poses if poses is not None else None)

    def _assert_flags_match(self, all_frames, where: str) -> None:
        """2c-8 shadow: with the carrier present, the flag-selected set must
        equal the CURRENT working set at every stage boundary — every
        permanent removal is write-through-recorded at surgery time, so any
        divergence is a missed or phantom drop. No-op on the legacy path."""
        db = getattr(self.ledger, "_db", None)
        if db is None:
            return
        for stream in db.streams:
            sel = [f.timestamp for f in stream.frames if f.is_selected_to_use]
            got = [f.timestamp for f in all_frames.get(stream.camera_id, [])]
            if got != sel:
                only_got = sorted(set(got) - set(sel))[:3]
                only_sel = sorted(set(sel) - set(got))[:3]
                raise AssertionError(
                    f"2c-8 shadow {where} cam{stream.camera_id}: "
                    f"{len(got)} in working set vs {len(sel)} flagged; "
                    f"working-only={only_got} flagged-only={only_sel} "
                    f"order_mismatch={set(got) == set(sel)}")

    def run(
        self,
        all_frames: dict[int, list[FrameObservations]],
        image_sizes: dict[int, tuple[int, int]],
        all_frames_full_rate: dict[int, list[FrameObservations]] | None = None,
        db=None,
    ) -> CalibrationResult | None:
        """Run the full calibration pipeline.

        Args:
            all_frames: Per-camera frame observations (subsampled for optimization).
            image_sizes: Per-camera image sizes {cam_id: (width, height)}.
            all_frames_full_rate: Optional full-rate frames for trajectory alignment.
                If None, uses all_frames for alignment too.

        Returns:
            CalibrationResult or None if calibration fails.
        """
        state = PipelineState()
        state.timer = Timer()
        camera_ids = sorted(all_frames.keys())
        # 2c-14: the carrier enters by the FRONT DOOR. An explicitly passed
        # db becomes the ledger's write-through target; the ledger-constructed
        # route (driver builds SelectionLedger(db=...)) keeps working during
        # the transition and must agree when both are given.
        if db is not None:
            if self.ledger._db is None:
                self.ledger._db = db
            else:
                assert self.ledger._db is db, "two different carriers passed"
        self._assert_flags_match(all_frames, "at run() entry")
        # 2c-14 shadow: image_sizes is derivable from the carrier's stream
        # resolutions — proven equal here before the parameter is retired.
        _db14 = getattr(self.ledger, "_db", None)
        if _db14 is not None:
            for _s in _db14.streams:
                _res = _s.info.resolution
                _got = image_sizes.get(_s.camera_id)
                if _res is not None and _got is not None:
                    assert tuple(_got) == tuple(_res), (
                        f"2c-14 shadow cam{_s.camera_id}: image_sizes "
                        f"{_got} vs stream resolution {_res}")

        if len(camera_ids) == 0:
            logger.error("No cameras to calibrate")
            return None

        # Determine reference camera
        ref_cam = self._config.reference_camera
        if ref_cam is None:
            ref_cam = max(camera_ids, key=lambda c: len(all_frames.get(c, [])))
        logger.info(f"Reference camera: {ref_cam}")

        # ================================================================
        # Stage 1: Per-camera initialization (OpenCV)
        # ================================================================
        n_obs_total = sum(len(f.observations) for fr in all_frames.values()
                          for f in fr)
        with state.timer.section("stage1_init",
                                  n_cameras=len(camera_ids),
                                  n_observations=n_obs_total):
            self._stage1_init(all_frames, image_sizes, camera_ids, state)

        active = [c for c in camera_ids if c in state.intrinsics]
        if not active:
            logger.error("No cameras initialized successfully")
            return None

        # Apply per-tag corner subsampling to observations fed into
        # Stages 2-4. Stage 1 already ran on the full set, so PnP init
        # and intrinsic init used all corners.
        # Save full-corner frames for PnP reinit (better pose init).
        all_frames_full_corners = all_frames  # reference before subsampling
        self._n_detected = {cid: sum(len(f.observations) for f in fr)
                            for cid, fr in all_frames.items()}
        mode = self._config.corner_subsample
        from mvis.detection.charuco_board import CharucoBoard
        if mode != "all" and isinstance(self._board, CharucoBoard):
            # ChArUco ids are chessboard corners, not tag * 4 + corner: nothing to thin
            logger.warning(f"corner_subsample '{mode}' has no meaning on a ChArUco board; "
                           "using 'all'")
            mode = "all"

        # Auto-fallback: if the requested mode thins observations
        # (top_left, bl, br, tr) but the detection is sparse, override
        # to "all" to avoid under-constrained calibration. Threshold:
        # median tags/frame < 10 for a board with ≥ 36 tags.
        num_expected_tags = (
            getattr(self._board, "_num_tags", 0) or self._board.num_points()
        )
        corners_per_tag = 4 if hasattr(self._board, "_num_tags") and \
            self._board._num_tags > 0 else 1
        _AUTO_FALLBACK_MIN_TAGS = 10

        if mode != "all" and num_expected_tags >= 20:
            tag_counts = []
            for frames in all_frames.values():
                for f in frames:
                    if corners_per_tag > 1:
                        tag_counts.append(
                            len({o.point_id // corners_per_tag
                                 for o in f.observations}))
                    else:
                        tag_counts.append(len(f.observations))
            if tag_counts:
                import statistics
                median_tags = statistics.median(tag_counts)
                if median_tags < _AUTO_FALLBACK_MIN_TAGS:
                    logger.warning(
                        f"Sparse detection (median {median_tags:.0f} tags/frame "
                        f"< {_AUTO_FALLBACK_MIN_TAGS}): overriding "
                        f"corner_subsample '{mode}' → 'all'"
                    )
                    mode = "all"

        # Board shape gives every corner its own height: a corner the subsample drops has no
        # data and stays at its prior (half the board under two_anti). Every corner is opt-in
        # (corner_subsample: all) because it doubles the observations and makes the solve about 1.8x slower (plan M4.3).
        if mode != "all" and self._config.board_heights:
            logger.warning(
                f"estimate.board_points with corner_subsample '{mode}': the corners it drops "
                f"keep their prior heights; set corner_subsample: all to estimate every corner")

        if mode != "all":
            from mvis.detection.corner_subsample import filter_frames
            before = sum(len(f.observations) for fr in all_frames.values() for f in fr)
            all_frames = filter_frames(all_frames, mode)
            after = sum(len(f.observations) for fr in all_frames.values() for f in fr)
            logger.info(
                f"Corner subsample ({mode}): "
                f"{before} → {after} observations "
                f"({100.0 * after / max(before, 1):.1f}%)"
            )

        # Detection-quality gating: unified quality gate applies all
        # configured filters (min tags, pooled-pose, corner views) and
        # reinitialises PnP poses with reprojection error check.
        from mvis.detection.quality_gate import (
            QualityGateConfig, apply_quality_gates,
        )
        with state.timer.section("quality_gate_stage1_2"):
            gate_cfg = QualityGateConfig(
                min_tag_count=self._config.min_tag_count,
                min_point_count=self._config.min_point_count,
                min_hull_fraction=self._config.min_hull_fraction,
                enable_pooled_pose_filter=self._config.enable_pose_filter,
                min_pose_tag_fraction=self._config.min_pose_tag_fraction,
                min_pose_tag_floor=self._config.min_pose_tag_floor,
                pose_sync_tolerance_s=self._config.pose_sync_tolerance_s,
                min_corner_views=(
                    self._config.min_corner_views
                    if self._config.enable_corner_observability_filter else 0),
                min_corner_views_per_camera=self._config.min_corner_views_per_camera,
                reinit_board_poses=True,
                max_pnp_reprojection_px=self._config.max_pnp_reprojection_px,
                pnp_gate_radtan_only=True,     # still the Stage 1 seed, as above
                pnp_method=self._config.pnp_method,
                log_tag_coverage=True,
            )
            _pre_gate = {c: list(fr) for c, fr in all_frames.items()}
            # ^ list-copies: apply_quality_gates MUTATES its input dict in
            # place, so a bare alias sees before==after and the diff
            # records nothing (the 2c-8 shadow caught exactly this)
            all_frames, new_poses, gate_result = apply_quality_gates(
                all_frames, self._board,
                state.intrinsics, state.dist_coeffs,
                state.distortion_models, gate_cfg,
                reference_camera=ref_cam,
                pnp_frames=all_frames_full_corners if mode != "all" else None,
            )
            self.ledger.diff(_pre_gate, all_frames, "stage1_gate", "quality_gate")
            state.board_poses = new_poses
            state.pose_by_ts = {
                cid: {f.timestamp: p
                      for f, p in zip(all_frames.get(cid, []), plist)}
                for cid, plist in new_poses.items()}
            self.pose_filter_stats = gate_result.pose_filter_stats
            self.corner_filter_stats = gate_result.corner_filter_stats
            self.quality_gate_result = gate_result

        # ================================================================
        # RS cameras: read from config
        # ================================================================
        for cam_cfg in (self._config.camera_configs or []):
            cid = getattr(cam_cfg, "camera_id", None)
            if cid is not None and shutter.is_rolling(
                    getattr(cam_cfg, "shutter_type", "auto")):
                state.rs_cameras[cid] = 0.05  # default prior sigma
                logger.info(f"  Camera {cid}: rolling shutter (from config, sigma=0.05)")

        # ================================================================
        # Stage 2 pre-filter: drop frames whose reprojection residual
        # against Stage 1's calibration is large. Radtan-only.
        # Timed separately so we can train a dedicated predictor and
        # see its cost in the per-stage summary.
        if self._config.stage2_pre_filter:
            n_radtan_cams = sum(1 for cid in active
                                if state.distortion_models.get(cid) == "radtan")
            n_input = sum(len(f) for f in all_frames.values())
            with state.timer.section("stage2_pre_filter",
                                      n_cameras=len(active),
                                      n_radtan_cameras=n_radtan_cams,
                                      n_frames_input=n_input):
                self._stage2_apply_pre_filter(all_frames, active, state)
            self._assert_flags_match(all_frames, "after stage2_pre_filter")

        # Stage 2: Per-camera GTSAM optimization (now includes RS if configured)
        # ================================================================
        n_obs_filtered = sum(len(f.observations) for fr in all_frames.values()
                             for f in fr)
        with state.timer.section("stage2_single_opt",
                                  n_cameras=len(active),
                                  n_observations=n_obs_filtered,
                                  corner_subsample=mode):
            self._stage2_single_cam_optimize(all_frames, image_sizes, active, state)
        self._assert_flags_match(all_frames, "after stage2_single_opt")
        self._sync_pose_identity(all_frames, state, "after stage2_single_opt")

        # Single-camera case: return directly (Stage 2 now has RS)
        if len(active) == 1:
            cam_id = active[0]
            single_result = state.single_cam_results.get(cam_id)
            self._finish_corner_accounting(single_result)
            _log_kb_valid_domain(single_result, image_sizes)
            timing_dict = state.timer.to_dict()
            logger.info(
                "Pipeline timing: "
                + ", ".join(f"{s['name']}={s['seconds']:.1f}s"
                            for s in timing_dict["sections"])
                + f", total={timing_dict['total_seconds']:.1f}s"
            )
            if single_result is not None:
                try:
                    single_result.timings = timing_dict
                except AttributeError:
                    pass
            return single_result

        # ================================================================
        # Stage 3: Trajectory alignment
        # Use full-rate frames if available (better angular velocity signal)
        # ================================================================
        # 2c-7 flip + batch-B retirement: with the carrier present the
        # alignment set is DEFINED by the flags — every frame whose exclusion
        # is absent or non-detection-tier (driver/pipeline drops happened
        # after this hand-off) — and its OBJECTS come from the carrier too:
        # when the driver stops passing all_frames_full_rate, the frames are
        # materialized by bridge-out (byte-identical uv by the T0 proofs;
        # resolution equality shadow-verified in 2c-14). The parameter
        # survives only for the legacy-JSON path (db=None).
        _db = getattr(self.ledger, "_db", None)
        if _db is not None:
            _src = all_frames_full_rate
            if _src is None:
                from mvis.mvisdb.bridge import frame_observations_from_db
                _src = {s.camera_id: frame_observations_from_db(_db, s.camera_id)
                        for s in _db.streams if s.frames}
            _flagged = {}
            for _cid, _fr in _src.items():
                _sel = {f.timestamp for f in _db.camera(_cid).frames
                        if f.exclude_reason is None
                        or not f.exclude_reason.startswith("detection:")}
                _flagged[_cid] = [f for f in _fr if f.timestamp in _sel]
            align_frames = _flagged
        else:
            align_frames = (all_frames_full_rate if all_frames_full_rate
                            else all_frames)
        n_frames_ref = len(align_frames.get(ref_cam, []))
        n_frames_nonref = sum(len(f) for cid, f in align_frames.items() if cid != ref_cam)
        with state.timer.section("stage3_alignment",
                                  n_frames_total=n_frames_ref + n_frames_nonref,
                                  n_frames_ref=n_frames_ref,
                                  n_frames_nonref=n_frames_nonref):
            self._stage3_trajectory_alignment(align_frames, active, ref_cam, state,
                                                image_sizes=image_sizes)

        if not state.extrinsics and len(active) > 1:
            logger.error("Trajectory alignment failed for all cameras")
            return None

        # Stage 1/2 gates ran per camera; re-pair before the joint solve
        all_frames, state.board_poses = self._drop_unpaired_nonref(
            all_frames, state.board_poses, ref_cam, "stage4_entry")

        # ================================================================
        # Stage 4: Joint multi-camera optimization
        # ================================================================
        # If adaptive subsample is enabled, upgrade HIGH-QUALITY frames
        # with more corners for Stage 4 joint optimization.
        # Only frames with >= 85% tags detected get all corners (excellent).
        # Frames with >= 70% get two_diag (good). Others stay top_left.
        # This enhances well-detected frames without amplifying noise
        # from sparse/oblique frames.
        frames_for_joint = all_frames
        if self._config.adaptive_subsample and mode != "all":
            from mvis.detection.corner_subsample import upgrade_high_quality_frames
            num_tags = getattr(self._board, "_num_tags", 36) or 36
            frames_for_joint = upgrade_high_quality_frames(
                all_frames, all_frames_full_corners, mode,
                num_tags=num_tags,
            )
            n_obs_adaptive = sum(len(f.observations) for fr in frames_for_joint.values() for f in fr)
            logger.info(f"Stage 4 adaptive: {n_obs_filtered} → {n_obs_adaptive} observations")

        n_obs_ref = sum(len(f.observations) for f in frames_for_joint.get(ref_cam, []))
        n_obs_nonref = sum(len(f.observations) for cid, fr in frames_for_joint.items()
                           if cid != ref_cam for f in fr)
        n_obs_joint = n_obs_ref + n_obs_nonref
        n_frames_ref_joint = sum(1 for f in frames_for_joint.get(ref_cam, [])
                                  if f.num_points >= self._board.min_points_for_pose())
        # 2c-13 shadow: the joint solve's working set must equal the flags.
        # Suspicion to test: the stage-3 re-gate records align-set drops as
        # permanent, but its output does NOT propagate here (local rebind in
        # _stage3_trajectory_alignment) — if it ever drops a solve-set frame,
        # this fires and the recording/propagation question becomes concrete.
        self._assert_flags_match(frames_for_joint, "at stage4 entry")
        with state.timer.section("stage4_joint_opt",
                                  n_cameras=len(active),
                                  n_observations=n_obs_joint,
                                  n_obs_ref=n_obs_ref,
                                  n_obs_nonref=n_obs_nonref,
                                  n_frames_ref=n_frames_ref_joint):
            result = self._stage4_joint_optimize(
                frames_for_joint, image_sizes, active, ref_cam, state)

        if result is not None:
            _log_stage_comparison("Stage 2 -> Stage 4 comparison",
                                  state.single_cam_results, result)

        # Detect diverged board poses after Stage 4 (ref camera poses).
        # Criteria: board z-depth in camera frame outside [0.05, 20]m.
        # These are mapped to frame timestamps and removed before Stage 5
        # rebuild so the factor graph is reconstructed without them.
        diverged_s5_timestamps = set()
        if result is not None and result.board_poses_gtsam:
            n_diverged = 0
            for bi, pose in enumerate(result.board_poses_gtsam):
                if pose is None:
                    continue
                T_board_cam = pose.inverse()
                z = T_board_cam.translation()[2]
                if z < _BOARD_Z_RANGE[0] or z > _BOARD_Z_RANGE[1]:
                    n_diverged += 1
                    # Map B-variable index to timestamp for frame removal
                    if bi < len(result.board_pose_times):
                        diverged_s5_timestamps.add(result.board_pose_times[bi])
            if n_diverged > 0:
                logger.warning(
                    f"Stage 4: {n_diverged}/{len(result.board_poses_gtsam)} "
                    f"board poses diverged (z outside [0.05, 20]m) — "
                    f"will be removed before Stage 5 rebuild")

        # Store first-pass result for external consumers (e.g. ablation script
        # that wants to report pass1 vs pass2).
        self.first_pass_result = result
        self.first_pass_frames = frames_for_joint
        pass1_result, rerun_input = result, None

        # ================================================================
        # Stage 5: Outlier-driven refinement
        # ================================================================
        if self._config.rerun_after_outlier_rejection and result is not None \
                and len(active) >= 1:
            # Remove diverged-pose frames from the ref camera before Stage 5.
            # solve() skips frames with None poses, so we set the pose to None
            # for frames whose timestamps match diverged B-variables.
            frames_for_rerun = frames_for_joint
            if diverged_s5_timestamps:
                import copy
                frames_for_rerun = copy.copy(all_frames)
                ref_frames = frames_for_rerun.get(ref_cam, [])
                n_removed = 0
                for fi, frame in enumerate(ref_frames):
                    if frame.timestamp in diverged_s5_timestamps:
                        self.ledger.record(ref_cam, frame.timestamp,
                                           "stage5_blank", "diverged_s5")
                        # Mark as removed by clearing observations
                        # (the frame stays in the list to preserve indexing,
                        # but quality_gate and solve() will skip it)
                        ref_frames[fi] = FrameObservations(
                            camera_id=ref_cam,
                            timestamp=frame.timestamp,
                            observations=[],
                            image_size=frame.image_size,
                        )
                        n_removed += 1
                if n_removed > 0:
                    logger.info(
                        f"Removed {n_removed} diverged-pose frames from ref "
                        f"camera before Stage 5 rebuild")

            rerun_input = frames_for_rerun
            with state.timer.section("stage5_rerun"):
                result = self._rerun_after_outlier_rejection(
                    frames_for_rerun, image_sizes, active, ref_cam, state,
                    result,
                )

        # Final cleanup on the output result: mark diverged poses as None
        # so they don't appear in the HTML report's 3D plot or error analysis.
        # Four criteria:
        #   1. Identity matrices (from norm sanitizer)
        #   2. Board behind camera (z < 0.05m in camera frame)
        #   3. Board too far (z > 20m in camera frame)
        #   4. Translation jump: pose far from temporal neighbors
        if result is not None and result.board_poses:
            flagged = _diverged_pose_flags(result)
            n_bp = len(flagged)
            n_final_diverged = 0
            for i in range(n_bp):
                if flagged[i] and result.board_poses[i] is not None:
                    result.board_poses[i] = None
                    if i < len(result.board_poses_gtsam):
                        result.board_poses_gtsam[i] = None
                    n_final_diverged += 1
            if n_final_diverged > 0:
                logger.info(
                    f"Final cleanup: {n_final_diverged} diverged/identity/"
                    f"jump-outlier poses marked None")

        if result is not None:
            # after the rerun and final cleanup, so the count describes the result as returned
            from mvis.calibration.corner_accounting import count_corners, guarded
            if result is pass1_result:          # no rerun, or it was reverted
                result.corner_accounting = guarded(
                    count_corners, result, self.first_pass_frames, self._board, image_sizes)
            else:
                result.corner_accounting = guarded(
                    count_corners, result, self.second_pass_frames, self._board, image_sizes,
                    before_filter=rerun_input, after_filter=self.filter_kept_frames,
                    pass1=pass1_result, pass1_frames=self.first_pass_frames,
                    pass1_bad_poses=_diverged_pose_flags(pass1_result))
        self._finish_corner_accounting(result)
        _log_kb_valid_domain(result, image_sizes)

        # Log timing summary
        timing_dict = state.timer.to_dict()
        logger.info(
            "Pipeline timing: "
            + ", ".join(f"{s['name']}={s['seconds']:.1f}s"
                        for s in timing_dict["sections"])
            + f", total={timing_dict['total_seconds']:.1f}s"
        )
        # Attach timing data to the result so OutputWriter can serialize it.
        if result is not None:
            try:
                result.timings = timing_dict
            except AttributeError:
                pass  # CalibrationResult may not yet have a timings field

        # Optional: launch Rerun visualization
        if self._config.rerun_visualization and result is not None:
            self._launch_rerun(result, all_frames, image_sizes)

        return result

    def _finish_corner_accounting(self, result) -> None:
        """Close the count against what this run was given: corners thinned by
        `corner_subsample`, or in frames a gate or a diverged pose removed before
        the final solve, are `not_in_solve`."""
        from mvis.calibration.corner_accounting import not_in_solve
        for cid, a in (getattr(result, "corner_accounting", None) or {}).items():
            n = self._n_detected.get(cid)
            if n is not None:
                a["detected"] = int(n)
                a["not_in_solve"] = not_in_solve(a)

    def set_detection_images(self, detection_images: dict):
        """Store detection images for Rerun visualization.

        Call this after detection and before run() so that --rerun-viz
        can display thumbnails alongside camera poses.
        """
        self._detection_images = detection_images

    def _launch_rerun(self, result, all_frames, image_sizes):
        """Launch Rerun viewer to display calibration results."""
        import os
        if os.environ.get("MVIS_NO_VIZ", "0") == "1":
            return
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            return
        try:
            from mvis.analysis.rerun_viz import RerunVisualizer
            viz = RerunVisualizer("mvis", spawn=True)
            if viz.enabled:
                det_imgs = getattr(self, '_detection_images', None)
                viz.log_calibration_result(
                    result, all_frames, self._board, image_sizes,
                    detection_images=det_imgs)
                viz.log_text("Calibration complete")
                logger.info("Rerun visualization launched")
        except Exception as e:
            logger.warning(f"Rerun visualization failed: {e}")

    # ------------------------------------------------------------------
    # Stage 1: Per-camera initialization
    # ------------------------------------------------------------------
    def _stage1_init(self, all_frames, image_sizes, camera_ids, state):
        """Initialize intrinsics and board poses for each camera using OpenCV."""
        _stage_banner("Stage 1: Per-camera initialization (OpenCV)")
        initializer = Initializer(self._board)
        max_init = self._config.max_init_frames

        for cid in camera_ids:
            frames = all_frames.get(cid, [])
            if len(frames) < 3:
                logger.warning(f"  Camera {cid}: too few frames ({len(frames)}), skipping")
                continue

            dm = self._config.distortion_models.get(cid, "radtan")
            img_size = image_sizes.get(cid, (640, 480))

            # Subsample for init (0 = use all frames). Strategy controlled
            # by init_frame_selection: "uniform" (time-spaced) or "diverse"
            # (position/size/orientation bins for better focal-length init).
            if max_init > 0 and len(frames) > max_init:
                from mvis.calibration.init_frame_selector import select_init_frames
                init_frames = select_init_frames(
                    frames, max_init, img_size,
                    strategy=self._config.init_frame_selection)
            else:
                init_frames = frames

            # Filter frames with too few tags before intrinsic init.
            # Stage 1 uses ALL corners so tag-based filter is sufficient.
            _min_tc = self._config.min_tag_count
            if _min_tc > 0:
                from mvis.detection.frame_quality import filter_frames_by_min_tags
                corners_per_tag = 4 if hasattr(self._board, "_num_tags") and \
                    self._board._num_tags > 0 else 1
                init_frames = filter_frames_by_min_tags(
                    init_frames, min_tags=_min_tc,
                    corners_per_tag=corners_per_tag)
                if len(init_frames) < 3:
                    logger.warning(f"  Camera {cid}: too few frames after min-tag filter "
                                   f"({len(init_frames)}), skipping")
                    continue

            seed_cfg = next((c for c in (self._config.camera_configs or [])
                             if getattr(c, "camera_id", None) == cid
                             and getattr(c, "initial_intrinsics", None)), None)
            if seed_cfg is not None:
                intr = np.asarray(seed_cfg.initial_intrinsics, float)
                dist = np.asarray(seed_cfg.initial_distortion
                                  if getattr(seed_cfg, "initial_distortion", None)
                                  else _fallback_dist(dm), float)
                if dm in KB_THETA_MODELS:
                    # pad a short seed; raise on non-zero extras (never truncate)
                    dist = np.asarray(canonical_dist(dm, dist), float)
                elif len(dist) < n_dist(dm):
                    # legacy short seed (e.g. 4-value omni [xi,k1,k2,p1]) -> p2 = 0
                    dist = np.pad(dist, (0, n_dist(dm) - len(dist)))
                logger.info(f"  Camera {cid}: SEEDED init fx={intr[0]:.1f} "
                            f"fy={intr[1]:.1f} (config initial_intrinsics)")
            else:
                logger.info(f"  Camera {cid}: intrinsics ({len(init_frames)}/{len(frames)} frames, {dm})...")
                try:
                    intr, dist = initializer.init_intrinsics_opencv(init_frames, img_size, dm)
                except Exception as e:
                    logger.error(f"  Camera {cid}: init failed: {e}")
                    continue

            # Stage 1 intrinsic sanity check
            if self._config.stage1_intrinsic_validation:
                w, h = img_size
                sane = True
                if intr[0] <= 0 or intr[1] <= 0:
                    logger.warning(f"  Camera {cid}: Stage 1 produced non-positive focal length "
                                   f"(fx={intr[0]:.1f}, fy={intr[1]:.1f})")
                    sane = False
                elif abs(intr[0] / intr[1] - 1.0) > 0.25:
                    logger.warning(f"  Camera {cid}: Stage 1 fx/fy asymmetry "
                                   f"(fx={intr[0]:.1f}, fy={intr[1]:.1f}, ratio={intr[0]/intr[1]:.3f})")
                    sane = False
                elif not (0.1 * w < intr[2] < 0.9 * w and 0.1 * h < intr[3] < 0.9 * h):
                    logger.warning(f"  Camera {cid}: Stage 1 principal point outside image "
                                   f"(cx={intr[2]:.1f}, cy={intr[3]:.1f}, img={w}x{h})")
                    sane = False
                elif _stage1_dist_extreme(dm, dist):
                    logger.warning(f"  Camera {cid}: Stage 1 distortion coefficients extreme: {dist}")
                    sane = False
                if not sane:
                    f0 = max(w, h) / 2.0
                    logger.warning(f"  Camera {cid}: falling back to heuristic init "
                                   f"(fx=fy={f0:.1f}, cx={w/2:.1f}, cy={h/2:.1f})")
                    intr = np.array([f0, f0, w / 2.0, h / 2.0])
                    dist = _fallback_dist(dm)

            logger.info(f"  Camera {cid}: board poses ({len(frames)} frames)...")
            # a VP seed is no trusted reference for a non-radtan model (see the Stage 2 pre-filter):
            # its pixel error says more about the seed than the pose
            max_reproj = self._config.max_pnp_reprojection_px if dm == "radtan" else 0.0
            poses = initializer.init_board_poses(
                frames, intr, dist, dm,
                max_reprojection_px=max_reproj,
                pnp_method=self._config.pnp_method)

            state.intrinsics[cid] = intr
            state.dist_coeffs[cid] = dist
            state.board_poses[cid] = poses
            state.pose_by_ts[cid] = {f.timestamp: p
                                     for f, p in zip(frames, poses)}
            state.distortion_models[cid] = dm

            n_ok = sum(1 for p in poses if p is not None)
            logger.info(f"  Camera {cid}: fx={intr[0]:.1f} fy={intr[1]:.1f}, "
                         f"{n_ok}/{len(frames)} poses OK")

    # ------------------------------------------------------------------
    # Stage 2: Per-camera GTSAM optimization
    # ------------------------------------------------------------------
    def _prefilter_frames_against_stage1(
        self, cid, frames, intr, dist, board_poses, distortion_model,
        pose_by_ts=None,
    ):
        """Drop frames whose reprojection residual against Stage 1's
        calibration is large.

        Returns ``(kept_frames, kept_source_indices, drop_count, threshold)``.
        The explicit source indices let the caller apply the exact same
        selection to the parallel board-pose track.

        Only applies to radtan (OpenCV calibrateCamera produces a
        trusted reference). For VP-init models (equidistant, FOV, DS,
        EUCM), Stage 1's distortion is zeros / hardcoded defaults — not
        a trusted reference — so we early-return with all frames.

        Sanity gate: if the filter would drop > max_drop_fraction of
        frames, pass everything through with a warning (something is
        fundamentally wrong; don't make it worse).
        """
        if board_poses is None:
            raise ValueError(
                f"Camera {cid}: Stage 2 pre-filter has no board-pose track")
        if len(frames) != len(board_poses):
            raise ValueError(
                f"Camera {cid}: Stage 2 pre-filter frame/pose length mismatch "
                f"({len(frames)} frames, {len(board_poses)} poses)")

        identity_indices = list(range(len(frames)))

        if distortion_model != "radtan":
            logger.info(
                f"  Camera {cid}: Stage 2 pre-filter skipped "
                f"(model={distortion_model}, no trusted Stage 1 reference)"
            )
            return frames, identity_indices, 0, None

        if not frames:
            return frames, identity_indices, 0, None

        import cv2 as _cv2
        # Board points in board frame: dict {point_id: (x,y,z)}
        # Use the board's _points_3d (Nx3) and frame.point_ids to index.
        board_pts_3d = getattr(self._board, "_points_3d", None)
        if board_pts_3d is None:
            logger.warning(
                f"  Camera {cid}: Stage 2 pre-filter skipped — "
                f"board has no _points_3d (unsupported board type)")
            return frames, identity_indices, 0, None

        K = np.array([[intr[0], 0.0, intr[2]],
                      [0.0,     intr[1], intr[3]],
                      [0.0,     0.0,     1.0]], dtype=np.float64)
        dist_arr = np.asarray(dist, dtype=np.float64).ravel()

        # Per-frame median residual, only for frames with a Stage 1 PnP pose.
        per_frame_med = []
        per_frame_idx = []  # index into original `frames` list
        for fi, f in enumerate(frames):
            # 2c-8b: pose by frame IDENTITY when the map is available; the
            # validated parallel track (same length, checked above) otherwise.
            pose = pose_by_ts.get(f.timestamp) if pose_by_ts is not None \
                else board_poses[fi]
            if pose is None:
                continue

            # Initializer stores T_board_from_cam.  OpenCV projectPoints
            # requires T_cam_from_board, so invert before extracting R,t.
            T_cam_from_board = pose.inverse()
            R_cb = T_cam_from_board.rotation().matrix()
            t_cb = np.array(T_cam_from_board.translation()).ravel()
            # Project this frame's observed corners through K, dist using
            # cv2.projectPoints. Only use corners we observed.
            pts_3d = []
            obs_2d = []
            for obs in f.observations:
                pid = int(obs.point_id)
                if pid < 0 or pid >= len(board_pts_3d):
                    continue
                pts_3d.append(board_pts_3d[pid])
                obs_2d.append(obs.pixel)
            if len(pts_3d) < 4:
                continue
            pts_3d_np = np.asarray(pts_3d, dtype=np.float64).reshape(-1, 1, 3)
            obs_2d_np = np.asarray(obs_2d, dtype=np.float64)
            rvec, _ = _cv2.Rodrigues(R_cb)
            tvec = t_cb.reshape(3, 1)
            proj, _ = _cv2.projectPoints(pts_3d_np, rvec, tvec, K, dist_arr)
            proj = proj.reshape(-1, 2)
            residuals = np.linalg.norm(proj - obs_2d_np, axis=1)
            if len(residuals) == 0:
                continue
            per_frame_med.append(float(np.median(residuals)))
            per_frame_idx.append(fi)

        if len(per_frame_med) < 5:
            return frames, identity_indices, 0, None

        meds = np.asarray(per_frame_med)
        # Threshold = median + 3·MAD (robust to the few catastrophic PnP
        # failures that would dominate p99). MAD = median absolute
        # deviation; for Gaussian noise, 1.4826·MAD ≈ stddev. So
        # median + 3·MAD ≈ 99.7th-percentile-equivalent in a robust way
        # that doesn't get inflated by extreme outliers. Lower-bounded
        # at min_threshold_px (default 2 px) so good-quality data isn't
        # over-trimmed.
        med_of_med = float(np.median(meds))
        mad = float(np.median(np.abs(meds - med_of_med)))
        threshold = max(
            float(self._config.stage2_pre_filter_min_threshold_px),
            med_of_med + 3.0 * 1.4826 * mad,
        )
        keep_mask = meds <= threshold
        n_drop = int((~keep_mask).sum())

        # Sanity gate: refuse to drop more than max_drop_fraction
        max_drop = int(self._config.stage2_pre_filter_max_drop_fraction
                       * len(frames))
        if n_drop > max_drop:
            logger.warning(
                f"  Camera {cid}: Stage 2 pre-filter would drop "
                f"{n_drop}/{len(frames)} frames "
                f"(>{100*self._config.stage2_pre_filter_max_drop_fraction:.0f}% cap) "
                f"— skipping filter, optimizer init may be unreliable")
            return frames, identity_indices, 0, None

        scored_indices = set(per_frame_idx)
        kept_scored_indices = {
            per_frame_idx[i] for i in range(len(meds)) if keep_mask[i]
        }
        # Frames we couldn't score (no PnP pose or <4 corners) are
        # passed through unchanged — don't drop on missing data.
        kept_source_indices = [
            i for i in identity_indices
            if i in kept_scored_indices or i not in scored_indices
        ]
        kept = [frames[i] for i in kept_source_indices]
        return kept, kept_source_indices, n_drop, threshold

    def _stage2_apply_pre_filter(self, all_frames, camera_ids, state):
        """Apply the Stage 2 input pre-filter to each camera in place.

        Mutates ``all_frames[cid]`` and ``state.board_poses[cid]`` with the
        same source indices, preserving their one-to-one identity. Radtan-only;
        other models
        skip with a logged reason. Rolling-shutter cameras also skip:
        Stage 1 is GS-only, so its residual is not a trustworthy
        outlier signal for an RS camera, and dropping frames removes
        geometric coverage the RS readout estimator needs (caused
        regressions on mvis_elp + tum_rgbd_bag, see
        historical regression notes). Called from the
        pipeline driver with its own timer section so cost shows up
        separately in the timing summary + training data.
        """
        _stage_banner("Stage 2 pre-filter: input frame quality gate")
        for cid in camera_ids:
            if cid not in state.intrinsics:
                continue
            if cid in state.rs_cameras:
                logger.info(
                    f"  Camera {cid}: Stage 2 pre-filter skipped "
                    f"(rolling shutter — Stage 1 init is GS-only)"
                )
                continue
            frames = all_frames.get(cid, [])
            poses = state.board_poses.get(cid)
            if poses is None:
                raise ValueError(
                    f"Camera {cid}: Stage 2 pre-filter has no board-pose track")
            if len(frames) != len(poses):
                raise ValueError(
                    f"Camera {cid}: Stage 2 pre-filter frame/pose length mismatch "
                    f"({len(frames)} frames, {len(poses)} poses)")
            dm = state.distortion_models.get(cid, "radtan")
            kept, kept_indices, n_drop, thr = self._prefilter_frames_against_stage1(
                cid,
                frames,
                state.intrinsics[cid],
                state.dist_coeffs[cid],
                poses,
                dm,
                pose_by_ts=state.pose_by_ts.get(cid),
            )
            if n_drop > 0:
                logger.info(
                    f"  Camera {cid}: Stage 2 pre-filter kept "
                    f"{len(kept)}/{len(all_frames[cid])} frames "
                    f"(threshold={thr:.2f} px, dropped {n_drop})"
                )
            if kept is not all_frames.get(cid):
                self.ledger.diff_one(cid, all_frames.get(cid, []), kept,
                                     "stage2_prefilter", "reproj_vs_stage1")
            all_frames[cid] = kept
            state.board_poses[cid] = [poses[i] for i in kept_indices]

    def _stage2_single_cam_optimize(self, all_frames, image_sizes, camera_ids, state):
        """Refine each camera's intrinsics and poses with GTSAM."""
        _stage_banner("Stage 2: Per-camera GTSAM optimization")
        initializer = Initializer(self._board)

        # Subsample per camera for Stage 2 — intrinsic calibration does not
        # need the full frame rate kept for trajectory alignment.  Frames and
        # their PnP poses form one parallel track, so select both with the same
        # explicit source indices and retain the full track for the Stage 4
        # handoff.
        stage2_hz = self._config.stage2_target_hz
        if stage2_hz is not None:
            from mvis.io.frame_selector import select_frame_indices_by_hz
            stage2_indices = select_frame_indices_by_hz(
                all_frames, target_hz=stage2_hz)
            stage2_frames = {
                cid: [frames[i] for i in stage2_indices[cid]]
                for cid, frames in all_frames.items()
            }
            # 2c-16: a stage-LOCAL subsample is not an exclusion (never
            # recorded), but it must be a pure subset of the flag-selected
            # set — anything else means it resurrected a dropped frame
            db16 = getattr(self.ledger, "_db", None)
            if db16 is not None:
                for cid16, fr16 in stage2_frames.items():
                    sel16 = {f.timestamp for f in db16.camera(cid16).frames
                             if f.is_selected_to_use}
                    assert all(f.timestamp in sel16 for f in fr16), (
                        f"2c-16: stage2 subsample resurrected frames cam{cid16}")
            for cid in camera_ids:
                n_before = len(all_frames.get(cid, []))
                n_after = len(stage2_frames.get(cid, []))
                if n_after < n_before:
                    logger.info(
                        f"  Camera {cid}: Stage 2 subsample "
                        f"{n_before} -> {n_after} frames")
        else:
            stage2_indices = {
                cid: list(range(len(frames)))
                for cid, frames in all_frames.items()
            }
            stage2_frames = {
                cid: list(frames) for cid, frames in all_frames.items()
            }

        full_pose_tracks = {}
        stage2_initial_poses = {}
        for cid in camera_ids:
            if cid not in state.intrinsics:
                continue
            full_frames = all_frames.get(cid, [])
            full_poses = state.board_poses.get(cid)
            if full_poses is None:
                raise ValueError(
                    f"Camera {cid}: Stage 2 has no board-pose track")
            if len(full_frames) != len(full_poses):
                raise ValueError(
                    f"Camera {cid}: Stage 2 frame/pose length mismatch "
                    f"({len(full_frames)} frames, {len(full_poses)} poses)")
            full_pose_tracks[cid] = list(full_poses)
            stage2_initial_poses[cid] = [
                full_poses[i] for i in stage2_indices[cid]
            ]

        opts = OptimizerOptions(
            max_iterations=self._config.max_iterations_single,
            pixel_sigma=self._config.pixel_sigma,
            huber_threshold=self._config.huber_threshold,
            lm_lambda_lower=self._config.lm_lambda_lower,
            lm_lambda_upper=self._config.lm_lambda_upper,
            verbose=self._config.verbose,
            rs_cameras=state.rs_cameras or None,
            board_heights=self._config.board_heights,
            board_height_sigma=self._config.board_height_sigma,
        )

        for cid in camera_ids:
            frames = stage2_frames.get(cid, [])
            if cid not in state.intrinsics:
                continue
            init_poses = stage2_initial_poses[cid]
            if len(frames) != len(init_poses):
                raise ValueError(
                    f"Camera {cid}: Stage 2 selected frame/pose length mismatch "
                    f"({len(frames)} frames, {len(init_poses)} poses)")

            dm = state.distortion_models[cid]
            img_size = image_sizes.get(cid, (640, 480))
            optimizer = Optimizer(self._board, opts)
            n_obs = sum(len(f.observations) for f in frames)

            with state.timer.section(f"camera_{cid}",
                                      n_observations=n_obs,
                                      n_frames=len(frames),
                                      distortion_model=dm):
                try:
                    # double_sphere / omni_radtan: the two-phase schedule
                    # (omni_schedule.solve_phased); P2 keeps every P1 option.
                    if dm in PHASED_MODELS:
                        result = solve_phased(
                            optimizer, frames, state.intrinsics[cid],
                            state.dist_coeffs[cid], init_poses, dm, img_size, cid)
                    else:
                        result = optimizer.solve_single_camera(
                            frames, state.intrinsics[cid], state.dist_coeffs[cid],
                            init_poses, dm, img_size, camera_id=cid)
                except Exception as e:
                    logger.error(f"  Camera {cid}: single-cam optimization failed: {e}")
                    continue

            state.single_cam_results[cid] = result

            # Update intrinsics and poses with optimized values
            cam_params = result.cameras[0]
            state.intrinsics[cid] = cam_params.intrinsics
            state.dist_coeffs[cid] = cam_params.distortion

            # Merge optimized subset poses back into the complete pre-Stage-2
            # timeline. valid_frame_indices is local to `frames`; compose it
            # with stage2_indices to recover the full-track position.  Frames
            # outside the subsample get a PnP re-init with the refined
            # intrinsics for Stage 3/4.
            if result.board_poses_gtsam:
                vfi = result.valid_frame_indices
                if not vfi:
                    raise ValueError(
                        f"Camera {cid}: optimizer returned poses without "
                        "valid_frame_indices")
                if len(vfi) != len(result.board_poses_gtsam):
                    raise ValueError(
                        f"Camera {cid}: optimizer result index/pose mismatch "
                        f"({len(vfi)} indices, "
                        f"{len(result.board_poses_gtsam)} poses)")
                full = all_frames.get(cid, [])
                selected_full_indices = stage2_indices[cid]
                merged_poses = list(full_pose_tracks[cid])
                # Selected frames: the optimized pose, or None where the frame
                # had no B-variable.
                for full_index in selected_full_indices:
                    merged_poses[full_index] = None
                for local_index, pose in zip(vfi, result.board_poses_gtsam):
                    if local_index < 0 or local_index >= len(selected_full_indices):
                        raise ValueError(
                            f"Camera {cid}: optimizer frame index {local_index} "
                            f"outside selected track of length "
                            f"{len(selected_full_indices)}")
                    merged_poses[selected_full_indices[local_index]] = pose
                # Frames outside the stage-2 subsample: a PnP re-init with the
                # refined intrinsics (keeps the 2c-9 lockstep).
                selected_set = set(selected_full_indices)
                rest_indices = [i for i in range(len(full))
                                if i not in selected_set]
                if rest_indices:
                    rest_poses = initializer.init_board_poses(
                        [full[i] for i in rest_indices],
                        state.intrinsics[cid], state.dist_coeffs[cid], dm,
                        max_reprojection_px=self._config.max_pnp_reprojection_px)
                    for i, p in zip(rest_indices, rest_poses):
                        merged_poses[i] = p
                state.board_poses[cid] = merged_poses

            if len(state.board_poses[cid]) != len(all_frames[cid]):
                raise ValueError(
                    f"Camera {cid}: Stage 2 handoff frame/pose length mismatch "
                    f"({len(all_frames[cid])} frames, "
                    f"{len(state.board_poses[cid])} poses)")

            err_stats = result.reprojection_errors[0] if result.reprojection_errors else None
            rs_str = ""
            if cam_params.rolling_shutter_readout != 0:
                rs_str = f" RS={cam_params.rolling_shutter_readout*1000:.2f}ms"
                state.rs_readouts[cid] = cam_params.rolling_shutter_readout
            if err_stats:
                logger.info(f"  Camera {cid}: median={err_stats.median:.2f}px "
                             f"P95={err_stats.p95:.2f}px, "
                             f"fx={cam_params.fx:.1f} fy={cam_params.fy:.1f}{rs_str}")
            else:
                logger.info(f"  Camera {cid}: fx={cam_params.fx:.1f} fy={cam_params.fy:.1f}{rs_str}")

            # --- Detect and remove diverged poses ---
            # First pass: detect, remove, quality-gate, reoptimize.
            # Subsequent: just detect and remove (no reopt — intrinsics
            # are already converged, remaining diverged poses are rare).
            from mvis.detection.frame_quality import (
                filter_frames_by_min_tags,
                filter_frames_dual,
                filter_observations_by_min_views_per_camera,
            )
            corners_per_tag = 4 if hasattr(self._board, "_num_tags") and \
                self._board._num_tags > 0 else 1

            diverged = _detect_diverged_poses_from_state(
                state.board_poses.get(cid, []), min_z=_BOARD_Z_RANGE[0], max_z=_BOARD_Z_RANGE[1])
            pose_track_length = len(state.board_poses.get(cid, []))
            logger.info(f"  Camera {cid}: diverged pose check: "
                        f"{len(diverged)}/{pose_track_length} flagged")

            if diverged and len(diverged) < 0.1 * pose_track_length:
                logger.warning(
                    f"  Camera {cid}: {len(diverged)} diverged poses, "
                    f"removing and reoptimizing")
                # Remove diverged frames
                cur_frames = all_frames.get(cid, frames)
                clean_frames = [f for i, f in enumerate(cur_frames)
                                if i not in diverged]
                # Re-apply quality gate (dual filter)
                clean_frames = filter_frames_dual(
                    clean_frames,
                    min_tag_count=self._config.min_tag_count,
                    min_point_count=self._config.min_point_count,
                    corners_per_tag=corners_per_tag)
                clean_dict = filter_observations_by_min_views_per_camera(
                    {cid: clean_frames},
                    min_views=self._config.min_corner_views_per_camera,
                    min_points_for_pose=self._board.min_points_for_pose())
                clean_frames = clean_dict.get(cid, clean_frames)

                if len(clean_frames) >= 3:
                    self.ledger.diff_one(cid, cur_frames, clean_frames,
                                         "stage2_diverged", "pose_diverged")
                    all_frames[cid] = clean_frames
                    cur_intr = state.intrinsics[cid]
                    cur_dist = state.dist_coeffs[cid]
                    clean_poses = initializer.init_board_poses(
                        clean_frames, cur_intr, cur_dist, dm,
                        max_reprojection_px=self._config.max_pnp_reprojection_px)
                    state.board_poses[cid] = clean_poses

                    # Rebuild graph and reoptimize (once)
                    try:
                        result_r = optimizer.solve_single_camera(
                            clean_frames, cur_intr, cur_dist, clean_poses,
                            dm, img_size, camera_id=cid)
                        state.single_cam_results[cid] = result_r
                        cam_r = result_r.cameras[0]
                        state.intrinsics[cid] = cam_r.intrinsics
                        state.dist_coeffs[cid] = cam_r.distortion
                        if result_r.board_poses_gtsam:
                            state.board_poses[cid] = poses_by_frame(
                                result_r, len(clean_frames), clean_poses)
                        # Check for remaining diverged (remove only, no reopt)
                        diverged2 = _detect_diverged_poses_from_state(
                            state.board_poses.get(cid, []),
                            min_z=_BOARD_Z_RANGE[0], max_z=_BOARD_Z_RANGE[1])
                        if diverged2:
                            logger.info(
                                f"  Camera {cid}: {len(diverged2)} remaining "
                                f"diverged after reopt, removing (no reopt)")
                            clean2 = [f for i, f in enumerate(
                                all_frames[cid]) if i not in diverged2]
                            self.ledger.diff_one(
                                cid, all_frames[cid], clean2,
                                "stage2_diverged", "pose_diverged_reopt")
                            all_frames[cid] = clean2
                            state.board_poses[cid] = [
                                p for i, p in enumerate(
                                    state.board_poses[cid])
                                if i not in diverged2]

                        # Report final error after all diverged removed
                        err_r = result_r.reprojection_errors[0] \
                            if result_r.reprojection_errors else None
                        if err_r:
                            logger.info(
                                f"  Camera {cid} reopt: "
                                f"median={err_r.median:.2f}px "
                                f"P95={err_r.p95:.2f}px, "
                                f"fx={cam_r.fx:.1f} fy={cam_r.fy:.1f} "
                                f"({len(all_frames[cid])} frames)")
                        else:
                            logger.info(
                                f"  Camera {cid} reopt: "
                                f"fx={cam_r.fx:.1f} fy={cam_r.fy:.1f} "
                                f"({len(all_frames[cid])} frames)")
                    except Exception as e:
                        logger.error(
                            f"  Camera {cid}: reoptimization failed: {e}")

    # ------------------------------------------------------------------
    # Stage 3: Trajectory alignment
    # ------------------------------------------------------------------
    def _stage3_trajectory_alignment(self, all_frames, camera_ids, ref_cam, state,
                                       image_sizes=None):
        """Align camera trajectories to find extrinsics, time offsets, and outliers.

        all_frames may be full-rate (not subsampled) for better alignment.
        Board poses are computed on-the-fly using optimized intrinsics from Stage 2.
        """
        _stage_banner(f"Stage 3: Trajectory alignment (ref=cam {ref_cam})")
        initializer = Initializer(self._board)

        # Compute board poses for alignment frames using Stage 2 intrinsics.
        # Apply quality gate: min 4 tags + PnP reprojection error check.
        align_poses = {}
        reuse = True
        for cid in camera_ids:
            if cid not in state.intrinsics:
                continue
            frames = all_frames.get(cid, [])
            if not frames:
                continue
            # 2c-10: reuse by IDENTITY, not frame count. The old check
            # ("same length => same data") silently applied solve-set poses
            # positionally to DIFFERENT align frames whenever counts
            # coincided; now reuse happens only when every align frame's
            # pose is identity-known (align set within the solve set), and
            # each frame gets ITS pose.
            p_map = state.pose_by_ts.get(cid)
            if p_map is not None and all(f.timestamp in p_map for f in frames):
                align_poses[cid] = [p_map[f.timestamp] for f in frames]
            else:
                reuse = False
                break

        if not reuse:
            from mvis.detection.quality_gate import (
                QualityGateConfig, apply_quality_gates,
            )
            gate_cfg = QualityGateConfig(
                min_tag_count=4,
                min_point_count=16,
                reinit_board_poses=True,
                max_pnp_reprojection_px=self._config.max_pnp_reprojection_px,
            )
            _pre_s3 = {c: list(fr) for c, fr in all_frames.items()}
            all_frames, align_poses, _ = apply_quality_gates(
                all_frames, self._board,
                state.intrinsics, state.dist_coeffs,
                state.distortion_models, gate_cfg,
                reference_camera=ref_cam,
            )
            self.ledger.diff(_pre_s3, all_frames, "stage3_gate", "quality_gate")

        try:
            extrinsics, time_offsets, outlier_frame_timestamps = \
                initializer.init_extrinsics_trajectory(
                    all_frames, align_poses, ref_cam,
                    image_sizes=image_sizes,
                    skip_temporal=not self._config.optimize_time_offset)
        except Exception as e:
            logger.warning(f"  Trajectory alignment failed: {e}")
            extrinsics, time_offsets, outlier_frame_timestamps = None, {}, {}

        # Validate temporal alignment: detect if cameras are hardware-synced.
        # ONLY override dt→0 when time-offset optimization is disabled
        # (no_td=True). When the user explicitly enables time-offset
        # calibration (optimize_time_offset=True), respect the cross-
        # correlation result even if the timestamps appear close — ROS-
        # bagged cameras share a system clock but may have real USB/
        # processing delays of 50-200ms.
        if extrinsics and time_offsets and not self._config.optimize_time_offset:
            for cid, dt in list(time_offsets.items()):
                ref_ts = np.array([f.timestamp for f in all_frames.get(ref_cam, [])])
                cam_ts = np.array([f.timestamp for f in all_frames.get(cid, [])])
                if len(ref_ts) > 0 and len(cam_ts) > 0:
                    n_matched = 0
                    for t in ref_ts[:min(100, len(ref_ts))]:
                        if np.min(np.abs(cam_ts - t)) < 0.05:
                            n_matched += 1
                    ratio = n_matched / min(100, len(ref_ts))
                    if ratio > 0.5 and abs(dt) > 0.05:
                        logger.info(f"  Camera {cid}: timestamps synced ({ratio:.0%} within 50ms), "
                                     f"overriding dt={dt:.4f}s → 0.0s")
                        time_offsets[cid] = 0.0
        elif extrinsics and time_offsets and self._config.optimize_time_offset:
            for cid, dt in time_offsets.items():
                if abs(dt) > 0.001:
                    logger.info(f"  Camera {cid}: keeping cross-corr dt={dt*1000:.1f}ms "
                                 f"(time-offset optimization enabled)")

        if not extrinsics:
            logger.info("  Falling back to co-visible frame alignment...")
            try:
                extrinsics = initializer.init_extrinsics(
                    all_frames, align_poses, ref_cam)
            except Exception as e:
                logger.error(f"  Extrinsic initialization failed: {e}")
                extrinsics = {}
            time_offsets = {}
            outlier_frame_timestamps = {}

        state.extrinsics = extrinsics or {}
        state.time_offsets = time_offsets
        state.outlier_frame_timestamps = outlier_frame_timestamps

        for cid in camera_ids:
            if cid == ref_cam or cid not in state.extrinsics:
                continue
            ext = state.extrinsics[cid]
            dt = state.time_offsets.get(cid, 0.0)
            n_outlier = len(
                state.outlier_frame_timestamps.get(cid, set()))
            logger.info(f"  Camera {cid}: dt={dt:.6f}s, "
                         f"outlier frames={n_outlier}")

    # ------------------------------------------------------------------
    # Stage 4: Joint multi-camera optimization
    # ------------------------------------------------------------------
    def _stage4_joint_optimize(self, all_frames, image_sizes, camera_ids,
                                ref_cam, state, *, max_iterations=None):
        """Joint optimization of all cameras in one factor graph."""
        _stage_banner("Stage 4: Joint multi-camera optimization")

        # Determine RS handling: free, or fixed at Stage 2 values
        rs_opt = state.rs_cameras or None
        fixed_rs = None
        if self._config.freeze_rs_cameras and state.rs_readouts:
            # Freeze RS for specified cameras, keep others free
            fixed_rs = {cid: val for cid, val in state.rs_readouts.items()
                        if cid in self._config.freeze_rs_cameras}
            # Remove frozen cameras from free RS set
            if rs_opt:
                rs_opt = {cid: v for cid, v in rs_opt.items()
                          if cid not in self._config.freeze_rs_cameras}
                if not rs_opt:
                    rs_opt = None

        opts = OptimizerOptions(
            max_iterations=max_iterations or self._config.max_iterations,
            pixel_sigma=self._config.pixel_sigma,
            huber_threshold=self._config.huber_threshold,
            lm_lambda_lower=self._config.lm_lambda_lower,
            lm_lambda_upper=self._config.lm_lambda_upper,
            verbose=self._config.verbose,
            optimize_time_offset=self._config.optimize_time_offset,
            rs_cameras=rs_opt,
            fixed_rs_cameras=fixed_rs,
            frozen_intrinsics_cameras=self._config.frozen_intrinsics_cameras,
            board_heights=self._config.board_heights,
            board_height_sigma=self._config.board_height_sigma,
            snap_sync_tolerance_s=self._config.pair_sync_tolerance_s,
        )
        optimizer = Optimizer(self._board, opts)

        result = optimizer.solve_multi_camera(
            all_frames, state.intrinsics, state.dist_coeffs,
            state.board_poses, state.extrinsics,
            state.distortion_models, ref_cam, image_sizes,
            time_offsets_init=state.time_offsets,
            outlier_frame_timestamps=state.outlier_frame_timestamps,
        )

        return result

    # ------------------------------------------------------------------
    # Stage 5: Outlier-driven refinement
    # ------------------------------------------------------------------
    def _rerun_after_outlier_rejection(
        self, all_frames, image_sizes, active, ref_cam, state, first_result,
    ):
        """Filter observations with residual > threshold, then rerun Stage 4.

        Warm-starts with the first pass's optimized intrinsics, extrinsics,
        time offsets and board poses so Stage 4 converges in just a few
        iterations on the cleaned data.
        """
        from mvis.calibration.outlier_rejection import (
            compute_per_observation_residuals, filter_outliers,
        )

        _stage_banner("Stage 5: Outlier-driven refinement")
        with state.timer.section("compute_residuals"):
            residuals = compute_per_observation_residuals(
                first_result, all_frames, self._board, image_sizes)
        with state.timer.section("filter_outliers"):
            filtered_frames, stats = filter_outliers(
                all_frames, residuals, self._board,
                absolute_threshold_px=self._config.outlier_threshold_px,
                percentile=self._config.outlier_percentile,
                frame_median_threshold_px=self._config.frame_median_threshold_px,
                frame_median_percentile=self._config.frame_median_percentile,
            )
            # a copy: the Stage 5 gates below rebind this dict's entries in place
            self.filter_kept_frames = dict(filtered_frames)
            self.ledger.diff(all_frames, filtered_frames,
                             "stage5_outlier", "residual_outlier")
            # Observation-level: a corner whose residual condemned it
            # becomes is_valid=False, carrying that residual. Frames
            # dropped whole are frame-level above; only surviving frames'
            # corner losses are recorded here.
            for cid, frames_before in all_frames.items():
                after_by_ts = {f.timestamp: f
                               for f in filtered_frames.get(cid, [])}
                res_cam = residuals.get(cid, [])
                for fi, fb in enumerate(frames_before):
                    fa = after_by_ts.get(fb.timestamp)
                    if fa is None:
                        continue
                    kept_pids = {o.point_id for o in fa.observations}
                    rlist = res_cam[fi] if fi < len(res_cam) else []
                    for oi, o in enumerate(fb.observations):
                        if o.point_id in kept_pids:
                            continue
                        rv = rlist[oi] if oi < len(rlist) else None
                        self.ledger.record_obs(
                            cid, fb.timestamp, o.point_id,
                            "stage5_outlier", "residual_outlier",
                            residual_px=rv)

        # Warm-start state with first-pass values so we only re-optimize
        # from a near-optimum (fewer iterations needed).
        # Build warm-start board poses per-camera from the ref trajectory.
        warm_intr = {cam.camera_id: cam.intrinsics.copy()
                     for cam in first_result.cameras}
        warm_dist = {cam.camera_id: cam.distortion.copy()
                     for cam in first_result.cameras}
        warm_dm = {cam.camera_id: cam.distortion_model
                   for cam in first_result.cameras}

        # Update pipeline state with warm-start values.
        state.intrinsics = warm_intr
        state.dist_coeffs = warm_dist
        state.distortion_models = warm_dm

        # Build Stage 4 optimized pose lookup by timestamp.
        # These are much better than fresh PnP estimates — jointly
        # optimized across all cameras and observations.
        s4_pose_by_time: dict[float, object] = {}
        if first_result.board_poses and first_result.board_pose_times:
            import gtsam as _gtsam
            for bp, t in zip(first_result.board_poses,
                             first_result.board_pose_times):
                if bp is not None and not np.allclose(bp, np.eye(4)):
                    s4_pose_by_time[round(t, 6)] = _gtsam.Pose3(bp)

        # Quality gate: min-tag filter + PnP reprojection gate.
        # PnP reinit is still enabled so the reprojection gate can reject
        # bad frames (PnP median error > threshold → pose = None).
        from mvis.detection.quality_gate import (
            QualityGateConfig, apply_quality_gates,
        )
        with state.timer.section("quality_gate_stage5"):
            gate_cfg = QualityGateConfig(
                min_tag_count=4,
                min_point_count=16,
                reinit_board_poses=True,
                max_pnp_reprojection_px=self._config.max_pnp_reprojection_px,
            )
            _pre_s5 = {c: list(fr) for c, fr in filtered_frames.items()}
            filtered_frames, new_poses, _ = apply_quality_gates(
                filtered_frames, self._board,
                state.intrinsics, state.dist_coeffs,
                state.distortion_models, gate_cfg,
                reference_camera=ref_cam,
            )
            self.ledger.diff(_pre_s5, filtered_frames, "stage5_gate", "quality_gate")
            filtered_frames, new_poses = self._drop_unpaired_nonref(
                filtered_frames, new_poses, ref_cam, "stage5_gate")

            # Warm-start: replace PnP poses with Stage 4 optimized poses
            # for frames that passed the reprojection gate. Frames rejected
            # by PnP (pose = None) stay rejected.
            n_warm = 0
            for cid in new_poses:
                frames = filtered_frames.get(cid, [])
                for fi, pose in enumerate(new_poses[cid]):
                    if pose is None:
                        continue  # PnP rejected — keep rejected
                    if fi < len(frames):
                        t_key = round(frames[fi].timestamp, 6)
                        s4_pose = s4_pose_by_time.get(t_key)
                        if s4_pose is not None:
                            new_poses[cid][fi] = s4_pose
                            n_warm += 1
            if n_warm > 0:
                logger.info(f"Stage 5 warm-start: {n_warm} board poses "
                            f"from Stage 4 (vs PnP reinit)")

            state.board_poses = new_poses

        # Reuse the pass-1 extrinsics and time offsets as warm-start rather
        # than rerunning Stage 3 — the filtered data has fewer observations
        # per frame so Stage 3 can actually regress the alignment quality.
        # The joint optimizer will refine them anyway.
        state.extrinsics.update(_warm_start_extrinsics(first_result))
        state.time_offsets = dict(first_result.time_offsets)
        # Keep Stage 1/RANSAC outlier timestamps from pass 1 — their stable
        # sample identity still applies to frames retained by Stage 5; dropped
        # frames no longer contribute factors.

        # Re-run joint optimization on filtered set
        with state.timer.section("stage4_rerun_opt",
                                  n_observations=sum(
                                      len(f.observations)
                                      for fr in filtered_frames.values()
                                      for f in fr)):
            try:
                result2 = self._stage4_joint_optimize(
                    filtered_frames, image_sizes, active, ref_cam, state,
                    max_iterations=self._config.max_iterations_rerun)
            except ValueError as e:
                # Outlier rejection removed too many frames — revert to pass-1
                logger.warning(f"Stage 5 rerun failed ({e}) — keeping pass-1 result")
                return first_result

        if result2 is not None:
            _log_stage_comparison("Stage 4 -> Stage 5 comparison (outlier rejection)",
                                  first_result, result2)

        # Log diff per camera
        for cid in active:
            s = stats.get(cid)
            if s is None:
                continue
            before = next((e for e in first_result.reprojection_errors
                           if e.camera_id == cid), None)
            after = next((e for e in (result2.reprojection_errors if result2 else [])
                          if e.camera_id == cid), None)
            if before and after:
                logger.info(
                    f"Rerun cam {cid}: obs {s.n_observations_before} -> {s.n_observations_after}, "
                    f"P95 {before.p95:.3f} -> {after.p95:.3f} px, "
                    f"Max {before.max:.2f} -> {after.max:.2f} px"
                )

        # Stash outlier stats for downstream report
        self.outlier_stats = stats
        self.second_pass_frames = filtered_frames

        # --------------------------------------------------------------
        # Pass-2 sanity guardrail
        # --------------------------------------------------------------
        # If the outlier rerun produced pathological intrinsics or
        # regressed quality badly, revert to the pass-1 result. This
        # catches cases like MIC02/03 where warm-started LM on filtered
        # observations slid to a local minimum with fx<0 (a sign-flipped
        # intrinsic alias). High-value defensive check — pass-2 is
        # *supposed* to be an improvement over pass-1, never a regression.
        # The revert is rig-wide on purpose: pass 2 is one joint solve
        # (shared board poses, extrinsics, time offsets), so splicing one
        # camera's pass-1 intrinsics into the pass-2 rig would pair them with
        # extrinsics they were never solved with. A per-camera revert would
        # need a re-solve and change behaviour for every model, so it is not
        # done.
        if result2 is not None and _pass2_is_pathological(
                pass1=first_result, pass2=result2):
            logger.warning(
                "Pass-2 sanity check failed — reverting to pass-1 result."
            )
            # The discarded pass's marks must not survive it —
            # without this, the overlay describes a rerun that never counted
            n_unmarked = self.ledger.unmark("stage5_")
            if n_unmarked:
                logger.info(f"  Reverted {n_unmarked} stage-5 selection marks "
                            "(pass-2 discarded)")
            return first_result
        # 2c-17: exit checkpoint — the rerun's final working set equals the
        # flags, completing invariant coverage entry -> stage-5 exit
        self._assert_flags_match(filtered_frames, "at stage5 exit")
        return result2


# ---------------------------------------------------------------------------
# Free-function camera-only API — the analog of
# calibration.pipeline_cam_imu.run_cam_imu_calibration[_multi].
# Same per-camera `cams`-dict input so the sim/programmatic camera-only and cam-IMU
# paths read the same; thin wrappers over CalibrationPipeline (board mode).
# ---------------------------------------------------------------------------
def run_cam_calibration_multi(cams, board, *, config=None):
    """Camera-only calibration over N cameras.

    `cams`: list of per-camera dicts {t_cam, ids_list, uv_list, image_size,
    is_fisheye (or model), camera_id?} — the SAME shape as
    run_cam_imu_calibration_multi's `cams` (a `cal_vec` seed, if present, is ignored:
    camera-only estimates intrinsics). `board`: the Board object. `config`:
    CamCalibConfig. Returns the CalibrationResult (or None)."""
    from mvis.core.config import CamCalibConfig
    from mvis.core.types import FrameObservations, Observation

    cfg = config or CamCalibConfig()
    all_frames: dict = {}
    image_sizes: dict = {}
    distortion_models: dict = {}
    for i, cam in enumerate(cams):
        cid = int(cam.get("camera_id", i))
        img = tuple(cam["image_size"])
        image_sizes[cid] = img
        distortion_models[cid] = (cam.get("model")
                                  or ("equidistant" if cam.get("is_fisheye") else "radtan"))
        frames = []
        for t, ids, uv in zip(cam["t_cam"], cam["ids_list"], cam["uv_list"]):
            ids = np.asarray(ids, int)
            uv = np.asarray(uv, float).reshape(-1, 2)
            obs = [Observation(point_id=int(pid), pixel=p, timestamp=float(t))
                   for pid, p in zip(ids, uv)]
            if obs:
                frames.append(FrameObservations(camera_id=cid, timestamp=float(t),
                                                observations=obs, image_size=img))
        all_frames[cid] = frames

    pc = PipelineConfig(
        max_iterations=cfg.max_iterations,
        max_iterations_single=cfg.max_iterations,
        pixel_sigma=cfg.sigma_px,
        huber_threshold=cfg.huber_px,
        lm_lambda_lower=getattr(cfg, "lm_lambda_lower", LM_LAMBDA_LOWER_DEFAULT),
        lm_lambda_upper=getattr(cfg, "lm_lambda_upper", LM_LAMBDA_UPPER_DEFAULT),
        corner_subsample=cfg.corner_subsample,
        optimize_intrinsics=cfg.optimize_intrinsics,
        optimize_time_offset=cfg.optimize_time_offset,
        max_init_frames=cfg.max_init_frames,
        init_frame_selection=cfg.init_frame_selection,
        reference_camera=cfg.reference_camera,
        rerun_after_outlier_rejection=cfg.rerun_outliers,
        distortion_models=distortion_models,
        board_heights=bool(getattr(cfg, "optimize_board_points", False)),
        verbose=False,
    )
    return CalibrationPipeline(board, pc).run(all_frames, image_sizes)


def run_cam_calibration(t_cam, ids_list, uv_list, board, image_size, *,
                        is_fisheye=False, model=None, camera_id=0, config=None):
    """Single-camera convenience wrapper over `run_cam_calibration_multi`."""
    cam = dict(camera_id=camera_id, t_cam=t_cam, ids_list=ids_list,
               uv_list=uv_list, image_size=image_size, is_fisheye=is_fisheye)
    if model:
        cam["model"] = model
    return run_cam_calibration_multi([cam], board, config=config)
