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
"""Unified GTSAM factor graph optimizer for camera calibration.

Handles single-camera and multi-camera calibration with the same solver.
Single-camera is the special case: one camera, identity extrinsic.

Factor graph variables:
    B(i)       — board pose Pose3 at each unique timestamp in the timeline
    K(cam_id)  — intrinsics Vector [fx, fy, cx, cy, dist...] of dim
                 4 + max(4, n_dist(model)) (8 for the 4-or-fewer-coeff models,
                 zero-padded)
    E(cam_id)  — extrinsic Pose3 per non-ref camera (T_ref_cam)
    D(cam_id)  — time offset Vector1 per non-ref camera
    R(cam_id)  — RS readout Vector1 per camera
    L(pid)     — board landmark Point3

For each observation from camera c at timestamp t, pixel row r:
    1. Corrected time: t_corr = t + td_c + (r / H) * tr_c
    2. Interpolate board pose: T_board = SLERP(B(i), B(i+1), lambda(t_corr))
    3. Camera pose: T_cam = T_board (ref camera) or E(c)^{-1} * T_board (non-ref)
    4. Project: pixel = project(T_cam^{-1}, landmark, K(c))
    5. Residual: pixel - measured
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import gtsam
from mvis.detection.board import Board
from mvis.calibration.factors.pose_interpolation import se3_interpolate as _se3_interpolate
from mvis.calibration.board_shape import board_height_gauge_weights
from mvis.calibration.factors.camera_intrinsics import _cal_to_vector
from mvis.calibration.factors.camera_projection import k_cal_vec
from mvis.calibration.factors.projection import make_projector
from mvis.calibration.lm_params import (
    LM_LAMBDA_LOWER_DEFAULT, LM_LAMBDA_UPPER_DEFAULT, make_lm_params)
from mvis.calibration.lm_rescue import rescue_if_stuck
from mvis.core.camera_models import check_model, legacy_model
from mvis.core.types import (
    CalibrationResult,
    CameraParams,
    ExtrinsicTransform,
    FrameObservations,
    ReprojectionErrorStats,
)

logger = logging.getLogger(__name__)


def _sym(char: str, index: int) -> int:
    return gtsam.symbol(char, index)


@dataclass
class OptimizerOptions:
    """Optimizer configuration."""

    max_iterations: int = 15
    pixel_sigma: float = 1.0
    huber_threshold: float = 1.345
    absolute_error_tol: float = 1e-8
    relative_error_tol: float = 1e-5
    # LM damping bounds (see `mvis.calibration.lm_params`): GTSAM's own are
    # 0 / 1e5. The upper bound 1e2 caps a stuck iteration's retry loop (one
    # try per decade from the lambda at the stall: 7 from GTSAM's initial
    # 1e-5); the lower bound is left at GTSAM's 0 so results stay bitwise.
    lm_lambda_lower: float = LM_LAMBDA_LOWER_DEFAULT
    lm_lambda_upper: float = LM_LAMBDA_UPPER_DEFAULT
    optimize_intrinsics: bool = True
    optimize_time_offset: bool = False
    dt_prior_sigma: float = 0.5  # loose prior — let optimizer find the right offset
    rs_cameras: dict = None  # cam_id -> readout_prior_sigma
    fixed_rs_cameras: dict = None  # cam_id -> fixed_readout_value (seconds)
    frozen_intrinsics_cameras: set = None  # cam_ids whose intrinsics are frozen
    frozen_xi_cameras: set = None  # cam_ids whose xi (K[4]) is frozen (omni_radtan)
    readout_prior_sigma: float = 0.01
    # Report the formal 1-sigma of each RS readout from the marginal covariance
    # (one extra sparse factorization after the solve). Off = sigma None.
    readout_sigma: bool = True
    # Board heights (since 2026-10-31): the board is modelled as
    # its pose + one out-of-plane height h_i per corner. x / y stay pinned (the
    # printed grid), h_i gets a zero-mean prior of board_height_sigma, and the
    # plane the heights share with the board poses (mean, two tilts) is removed
    # by a 3-row gauge factor. False = the board stays at its given shape: x, y
    # and z all pinned, nothing reported.
    board_heights: bool = False
    board_height_sigma: float = 0.003
    # Without a time offset, an observation whose nearest ref pose is within
    # this many seconds is projected at that pose (camera_projection.synced_time)
    # instead of interpolated at its own stamp. 0 = always interpolate.
    snap_sync_tolerance_s: float = 0.0
    verbose: bool = False


class Optimizer:
    """GTSAM-based unified optimizer for camera calibration."""

    def __init__(self, board: Board, options: OptimizerOptions | None = None):
        self._board = board
        self._options = options or OptimizerOptions()

    @property
    def board(self) -> Board:
        return self._board

    @property
    def options(self) -> OptimizerOptions:
        return self._options

    def _is_rs_camera(self, cam_id: int) -> bool:
        if self._options.rs_cameras and cam_id in self._options.rs_cameras:
            return True
        if self._options.fixed_rs_cameras and cam_id in self._options.fixed_rs_cameras:
            return True
        return False

    def _is_fixed_rs_camera(self, cam_id: int) -> bool:
        return (self._options.fixed_rs_cameras is not None
                and cam_id in self._options.fixed_rs_cameras)

    def _rs_prior_sigma(self, cam_id: int) -> float:
        if self._options.rs_cameras and cam_id in self._options.rs_cameras:
            return self._options.rs_cameras[cam_id]
        return self._options.readout_prior_sigma

    # ==================================================================
    # Unified solver
    # ==================================================================

    def solve(
        self,
        all_frames: dict[int, list[FrameObservations]],
        all_intrinsics: dict[int, np.ndarray],
        all_dist_coeffs: dict[int, np.ndarray],
        all_board_poses: dict[int, list[gtsam.Pose3 | None]],
        distortion_models: dict[int, str],
        reference_camera: int = 0,
        image_sizes: dict[int, tuple[int, int]] | None = None,
        extrinsics: dict[int, gtsam.Pose3] | None = None,
        time_offsets_init: dict[int, float] | None = None,
        outlier_frames: dict[int, set[int]] | None = None,
        frozen_distortion: bool = False,
        outlier_frame_timestamps: dict[int, set[float]] | None = None,
    ) -> CalibrationResult:
        """Unified solver for single-camera and multi-camera calibration.

        For single-camera: pass one camera in all_frames, extrinsics=None.
        For multi-camera: pass all cameras, extrinsics for non-ref cameras.

        All observations use the same factor type: interpolation-based projection
        with optional extrinsic, time offset, RS readout, and intrinsics.
        """
        image_sizes = image_sizes or {}
        extrinsics = extrinsics or {}
        time_offsets_init = time_offsets_init or {}
        if outlier_frames is not None and outlier_frame_timestamps is not None:
            raise ValueError(
                "Pass either legacy outlier_frames indices or "
                "outlier_frame_timestamps, not both")

        camera_ids = sorted(all_frames.keys())
        for cam_id in camera_ids:
            if cam_id not in all_board_poses:
                raise ValueError(
                    f"Camera {cam_id}: board-pose track is missing")
            n_frames = len(all_frames[cam_id])
            n_poses = len(all_board_poses[cam_id])
            if n_frames != n_poses:
                raise ValueError(
                    f"Camera {cam_id}: frame/pose length mismatch "
                    f"({n_frames} frames, {n_poses} poses)")

        # Stage 3 and Stage 4 may operate on differently filtered frame lists.
        # Raw timestamps are therefore the canonical outlier identity.  Keep
        # the old integer-index API as an explicit boundary conversion for
        # external callers, but never propagate those indices internally.
        normalized_outlier_timestamps: dict[int, set[float]] = {}
        if outlier_frames is not None:
            for cam_id, frame_indices in outlier_frames.items():
                if cam_id not in all_frames:
                    raise ValueError(
                        f"Outlier indices reference unknown camera {cam_id}")
                timestamps = normalized_outlier_timestamps.setdefault(
                    cam_id, set())
                for frame_idx in frame_indices:
                    if (isinstance(frame_idx, bool)
                            or not isinstance(frame_idx, (int, np.integer))):
                        raise TypeError(
                            "Legacy outlier_frames values must be integer "
                            f"indices, got {frame_idx!r}")
                    if frame_idx < 0 or frame_idx >= len(all_frames[cam_id]):
                        raise ValueError(
                            f"Camera {cam_id}: outlier frame index "
                            f"{frame_idx} is out of range")
                    timestamps.add(
                        float(all_frames[cam_id][frame_idx].timestamp))
        elif outlier_frame_timestamps is not None:
            for cam_id, frame_timestamps in outlier_frame_timestamps.items():
                if cam_id not in all_frames:
                    raise ValueError(
                        f"Outlier timestamps reference unknown camera {cam_id}")
                timestamps = normalized_outlier_timestamps.setdefault(
                    cam_id, set())
                for timestamp in frame_timestamps:
                    try:
                        if isinstance(timestamp, (bool, np.bool_)):
                            raise ValueError
                        normalized_timestamp = float(timestamp)
                    except (TypeError, ValueError, OverflowError):
                        raise ValueError(
                            f"Camera {cam_id}: invalid outlier timestamp "
                            f"{timestamp!r}") from None
                    if not np.isfinite(normalized_timestamp):
                        raise ValueError(
                            f"Camera {cam_id}: invalid outlier timestamp "
                            f"{timestamp!r}")
                    timestamps.add(normalized_timestamp)
        is_single = len(camera_ids) == 1
        use_dt = self._options.optimize_time_offset and not is_single

        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()

        # -- Noise models --
        pixel_noise = gtsam.noiseModel.Isotropic.Sigma(2, self._options.pixel_sigma)
        robust_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(self._options.huber_threshold),
            pixel_noise,
        )
        pixel_noise_outlier = gtsam.noiseModel.Isotropic.Sigma(
            2, self._options.pixel_sigma * 5.0)
        robust_noise_outlier = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(self._options.huber_threshold),
            pixel_noise_outlier,
        )

        def _noise_for(cam_id, timestamp):
            if timestamp in normalized_outlier_timestamps.get(cam_id, set()):
                return robust_noise_outlier
            return robust_noise

        # -- Board landmarks --
        if self._options.board_heights:
            # x / y pinned, height free with a zero-mean prior
            lm_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([1e-6, 1e-6, self._options.board_height_sigma]))
        else:
            lm_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1e-6)
        lm_keys = []
        lm_n_obs = np.zeros(self._board.num_points(), int)   # vision factors per corner, as solved
        for pid in range(self._board.num_points()):
            lm_key = _sym("L", pid)
            pt = self._board.point_3d(pid)
            gtsam_pt = gtsam.Point3(pt[0], pt[1], pt[2])
            initial.insert(lm_key, gtsam_pt)
            graph.addPriorPoint3(lm_key, gtsam_pt, lm_noise)
            lm_keys.append(lm_key)
        if self._options.board_heights:
            graph.add(_board_height_gauge_factor(
                lm_keys, np.asarray(self._board.all_points_3d(), float)))

        # -- Build ref camera pose timeline --
        # Pose nodes come ONLY from the reference camera.
        # Non-ref camera observations interpolate between nearest ref poses.
        ref_frames = all_frames.get(reference_camera, list(all_frames.values())[0])
        ref_poses = all_board_poses.get(reference_camera,
                                         list(all_board_poses.values())[0])

        bp_timeline_t = []  # sorted timestamps of board pose nodes
        bp_timeline_idx = []  # corresponding bp indices
        bp_source_frame_indices = []  # source frame index for each B variable
        bp_count = 0

        # Very-weak anchor prior on every B variable — guarantees GTSAM
        # elimination works even when non-ref camera observations don't
        # bracket all of the last few ref poses (orphan B variables).
        # Sigma is huge (1e3 m + 1e3 rad) so it does not affect the
        # optimization, it only keeps the information matrix non-singular.
        b_anchor_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1e3] * 6))

        for fi, (frame, pose) in enumerate(zip(ref_frames, ref_poses)):
            if pose is None or frame.num_points < self._board.min_points_for_pose():
                continue
            bp_key = _sym("B", bp_count)
            initial.insert(bp_key, pose)
            graph.addPriorPose3(bp_key, pose, b_anchor_noise)
            bp_timeline_t.append(frame.timestamp)
            bp_timeline_idx.append(bp_count)
            bp_source_frame_indices.append(fi)
            bp_count += 1

        if bp_count == 0:
            raise ValueError("No valid board poses from reference camera")

        bp_timeline_t = np.array(bp_timeline_t)

        # -- Per-camera variables --
        # Intrinsics as a Vector of dim n_k = 4 + max(4, n_dist): 8 for every
        # <=4-coeff model (short dists zero-padded, as before), wider otherwise
        from mvis.calibration.factors.custom_projections import CUSTOM_MODELS
        for cam_id in camera_ids:
            intr = all_intrinsics[cam_id]
            dist = all_dist_coeffs[cam_id]
            cal_vec = k_cal_vec(distortion_models[cam_id], intr, dist)
            n_k = len(cal_vec)
            cal_key = _sym("K", cam_id)
            initial.insert(cal_key, cal_vec)

            # Freeze distortion params for custom models during Phase 1
            # of sequential unfreezing. Uses a tight prior on distortion
            # while leaving intrinsics free to optimize.
            if frozen_distortion and distortion_models[cam_id] in CUSTOM_MODELS:
                if distortion_models[cam_id] == "omni_radtan":
                    # Omni: K = [fx, fy, cx, cy, xi, k1, k2, p1, p2]
                    # Phase 1: xi has moderate prior (prevents divergence from
                    # xi-fx coupling), k1/k2/p1/p2 frozen
                    freeze_sigmas = np.array(
                        [1e6] * 4                 # fx,fy,cx,cy: free
                        + [1.0]                   # xi: moderate prior (±1.0)
                        + [1e-6] * (n_k - 5))     # k1,k2,p1,p2: frozen
                else:
                    # Other custom models: freeze all of K[4:]
                    freeze_sigmas = np.array(
                        [1e6] * 4                 # fx,fy,cx,cy: free
                        + [1e-6] * (n_k - 4))     # distortion: frozen
                graph.addPriorVector(
                    cal_key, cal_vec,
                    gtsam.noiseModel.Diagonal.Sigmas(freeze_sigmas))

            # Freeze intrinsics (fx,fy,cx,cy) for specific cameras.
            # Distortion is also frozen. Used to diagnose joint-opt degradation.
            if (self._options.frozen_intrinsics_cameras
                    and cam_id in self._options.frozen_intrinsics_cameras):
                freeze_sigmas = np.full(n_k, 1e-6)   # fx,fy,cx,cy + distortion: frozen
                graph.addPriorVector(
                    cal_key, cal_vec,
                    gtsam.noiseModel.Diagonal.Sigmas(freeze_sigmas))
                logger.info(f"Camera {cam_id}: intrinsics FROZEN at "
                            f"fx={intr[0]:.2f} fy={intr[1]:.2f}")

            # Freeze only xi (K[4]) for omni_radtan Phase 2.
            # Keeps fx/fy/cx/cy and k1/k2/p1/p2 free to optimize (p2 always
            # free, as in Kalibr).
            if (self._options.frozen_xi_cameras
                    and cam_id in self._options.frozen_xi_cameras
                    and distortion_models.get(cam_id) == "omni_radtan"):
                freeze_sigmas = np.array(
                    [1e6] * 4                 # fx,fy,cx,cy: free
                    + [1e-6]                  # xi: frozen
                    + [1e6] * (n_k - 5))      # k1,k2,p1,p2: free
                graph.addPriorVector(
                    cal_key, cal_vec,
                    gtsam.noiseModel.Diagonal.Sigmas(freeze_sigmas))
                logger.info(f"Camera {cam_id}: xi FROZEN at {dist[0]:.4f}")

        # Extrinsics for non-ref cameras
        # Input extrinsics are T_cam_ref (from init_extrinsics_trajectory).
        # The factor expects T_ref_cam: T_board_cam = T_board_ref * T_ref_cam.
        for cam_id in camera_ids:
            if cam_id == reference_camera or is_single:
                continue
            ext = extrinsics.get(cam_id)
            if ext is None:
                continue
            ek = _sym("E", cam_id)
            initial.insert(ek, ext.inverse())  # T_cam_ref → T_ref_cam

        # Time offsets for non-ref cameras
        if use_dt:
            for cam_id in camera_ids:
                if cam_id == reference_camera:
                    continue
                dk = _sym("D", cam_id)
                dt_init = time_offsets_init.get(cam_id, 0.0)
                initial.insert(dk, np.array([dt_init]))
                graph.addPriorVector(
                    dk, np.array([dt_init]),
                    gtsam.noiseModel.Isotropic.Sigma(1, self._options.dt_prior_sigma))

        # RS readout per camera
        for cam_id in camera_ids:
            if self._is_rs_camera(cam_id):
                rk = _sym("R", cam_id)
                if self._is_fixed_rs_camera(cam_id):
                    # Freeze RS at known value from Stage 2 single-camera calibration
                    fixed_val = self._options.fixed_rs_cameras[cam_id]
                    initial.insert(rk, np.array([fixed_val]))
                    graph.addPriorVector(
                        rk, np.array([fixed_val]),
                        gtsam.noiseModel.Isotropic.Sigma(1, 1e-6))
                    logger.info(f"Camera {cam_id}: RS readout FIXED at {fixed_val*1000:.2f}ms")
                else:
                    initial.insert(rk, np.array([0.01]))
                    graph.addPriorVector(
                        rk, np.array([0.0]),
                        gtsam.noiseModel.Isotropic.Sigma(1, self._rs_prior_sigma(cam_id)))

        # -- Build projection factors --
        fn_refs = []  # prevent GC of CustomFactor closures
        n_factors = 0
        n_outlier_factors = 0
        factor_frames: dict[int, set] = {}

        n_total_obs = 0
        n_snapped = 0
        from mvis.calibration.factors.camera_projection import find_bracketing_poses, synced_time
        snap_tol = 0.0 if use_dt else self._options.snap_sync_tolerance_s
        for cam_id in camera_ids:
            frames = all_frames[cam_id]
            poses = all_board_poses[cam_id]
            is_ref = (cam_id == reference_camera) or is_single
            dist_model = distortion_models[cam_id]
            h_cam = image_sizes.get(cam_id, (640, 480))[1]

            cal_key = _sym("K", cam_id)
            ek = _sym("E", cam_id) if not is_ref else None
            dk = _sym("D", cam_id) if (use_dt and not is_ref) else None
            rk = _sym("R", cam_id) if self._is_rs_camera(cam_id) else None

            # For non-ref cameras with time offset, shift the lookup time
            # so that pose bracketing selects the correct pair. Without this,
            # |dt| > pose_spacing causes lambda > 1 (extrapolation → divergence).
            dt_bracket = 0.0
            if not is_ref and use_dt:
                dt_bracket = time_offsets_init.get(cam_id, 0.0)

            for fi, (frame, pose) in enumerate(zip(frames, poses)):
                if pose is None or frame.num_points < self._board.min_points_for_pose():
                    continue

                t = synced_time(frame.timestamp, bp_timeline_t, snap_tol)
                n_snapped += int(not is_ref and t != frame.timestamp)
                t_lookup = t + dt_bracket

                # The two ref camera poses bracketing this time; skip outside the timeline.
                bracket = find_bracketing_poses(t_lookup, bp_timeline_t, bp_timeline_idx)
                if bracket is None:
                    continue
                bp0_idx, bp1_idx, t0, t1 = bracket

                bp0_key = _sym("B", bp0_idx)
                bp1_key = _sym("B", bp1_idx)

                noise_fi = _noise_for(cam_id, float(frame.timestamp))
                if noise_fi is robust_noise_outlier:
                    n_outlier_factors += len(frame.observations)

                factor_frames.setdefault(cam_id, set()).add(float(frame.timestamp))
                for obs in frame.observations:
                    lm_key = _sym("L", obs.point_id)
                    factor, fn_ref = _make_unified_factor(
                        measured=obs.pixel,
                        board_point=self._board.point_3d(obs.point_id),
                        pixel_row=obs.row, image_height=h_cam,
                        time_meas=t, time_0=t0, time_1=t1,
                        noise_model=noise_fi,
                        bp0_key=bp0_key, bp1_key=bp1_key,
                        lm_key=lm_key, cal_key=cal_key,
                        ext_key=ek, td_key=dk, tr_key=rk,
                        distortion_model=dist_model,
                    )
                    fn_refs.append(fn_ref)
                    graph.add(factor)
                    n_total_obs += 1
                    lm_n_obs[obs.point_id] += 1

        # -- Optimize --
        params = make_lm_params(
            self._options.max_iterations,
            absolute_error_tol=self._options.absolute_error_tol,
            relative_error_tol=self._options.relative_error_tol,
            lambda_lower=self._options.lm_lambda_lower,
            lambda_upper=self._options.lm_lambda_upper,
            verbose=self._options.verbose)

        rs_cams = [c for c in camera_ids if self._is_rs_camera(c)]
        rs_msg = f", RS={rs_cams}" if rs_cams else ""
        outlier_msg = f", {n_outlier_factors} downweighted" if n_outlier_factors > 0 else ""
        snap_msg = (f", {n_snapped} non-ref frames at their synced ref pose"
                    if n_snapped else "")
        logger.info(f"Optimization: {len(camera_ids)} cams, {bp_count} ref poses, "
                     f"{n_total_obs} observations, {graph.size()} factors{rs_msg}{outlier_msg}"
                     f"{snap_msg}")

        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
        result_values = optimizer.optimize()
        # A poor seed, such as a fisheye camera's vanishing-point focal length, can leave LM still
        # descending at its iteration limit, or stopped at its damping cap; a stop that is not stuck
        # keeps its result bitwise. The report goes on the result, so a solve still stuck is flagged.
        result_values, optimizer, lm_rescue = rescue_if_stuck(
            graph, initial, result_values, optimizer, params,
            lambda values, lm_params: gtsam.LevenbergMarquardtOptimizer(graph, values, lm_params),
            at_iteration_limit=True)
        logger.info(f"Error: {graph.error(initial):.4f} -> {graph.error(result_values):.4f} "
                    f"({optimizer.iterations()} iterations)")

        # -- Extract results --
        result = self._extract_results(
            result_values, graph, camera_ids, reference_camera,
            all_frames, all_board_poses, distortion_models, image_sizes,
            bp_timeline_t, bp_timeline_idx, bp_source_frame_indices,
            bp_count, use_dt, is_single, snap_tol, factor_frames, lm_n_obs,
        )
        result.lm_rescue = lm_rescue
        return result

    # ==================================================================
    # Backward-compatible wrappers
    # ==================================================================

    def solve_single_camera(
        self,
        frames: list[FrameObservations],
        intrinsics: np.ndarray,
        dist_coeffs: np.ndarray,
        board_poses: list[gtsam.Pose3 | None],
        distortion_model: str = "radtan",
        image_size: tuple[int, int] = (0, 0),
        camera_id: int = 0,
        frozen_distortion: bool = False,
    ) -> CalibrationResult:
        """Optimize a single camera. Delegates to solve()."""
        return self.solve(
            all_frames={camera_id: frames},
            all_intrinsics={camera_id: intrinsics},
            all_dist_coeffs={camera_id: dist_coeffs},
            all_board_poses={camera_id: board_poses},
            distortion_models={camera_id: distortion_model},
            reference_camera=camera_id,
            image_sizes={camera_id: image_size},
            frozen_distortion=frozen_distortion,
        )

    def solve_multi_camera(
        self,
        all_frames, all_intrinsics, all_dist_coeffs, all_board_poses,
        extrinsics, distortion_models, reference_camera=0,
        image_sizes=None, time_offsets_init=None, outlier_frames=None,
        frozen_distortion: bool = False,
        outlier_frame_timestamps=None,
    ) -> CalibrationResult:
        """Optimize multi-camera system. Delegates to solve()."""
        return self.solve(
            all_frames=all_frames,
            all_intrinsics=all_intrinsics,
            all_dist_coeffs=all_dist_coeffs,
            all_board_poses=all_board_poses,
            distortion_models=distortion_models,
            reference_camera=reference_camera,
            image_sizes=image_sizes,
            extrinsics=extrinsics,
            time_offsets_init=time_offsets_init,
            outlier_frames=outlier_frames,
            frozen_distortion=frozen_distortion,
            outlier_frame_timestamps=outlier_frame_timestamps,
        )

    # ==================================================================
    # Result extraction
    # ==================================================================

    def _extract_results(
        self, result_values, graph, camera_ids, reference_camera,
        all_frames, all_board_poses, distortion_models, image_sizes,
        bp_timeline_t, bp_timeline_idx, bp_source_frame_indices,
        bp_count, use_dt, is_single,
        snap_tol: float = 0.0,
        factor_frames: dict | None = None,
        lm_n_obs: np.ndarray | None = None,
    ) -> CalibrationResult:
        """Extract CalibrationResult from optimized values."""

        # Formal readout uncertainty: one Marginals factorization, read only the
        # R(cam) blocks. Skipped when no camera is RS; a failure (indeterminate
        # system) leaves the sigmas None rather than failing the calibration.
        readout_sigmas: dict = {}
        rs_ids = [c for c in camera_ids if self._is_rs_camera(c)
                  and result_values.exists(_sym("R", c))]
        if rs_ids and getattr(self._options, "readout_sigma", True):
            try:
                marg = gtsam.Marginals(graph, result_values)
                for cam_id in rs_ids:
                    cov = marg.marginalCovariance(_sym("R", cam_id))
                    readout_sigmas[cam_id] = float(np.sqrt(max(cov[0, 0], 0.0)))
            except Exception as e:  # pragma: no cover - depends on the problem
                logger.warning(f"Readout sigma: marginals unavailable ({e})")

        # Cameras (intrinsics + RS readout)
        cameras = []
        for cam_id in camera_ids:
            cal_vec = result_values.atVector(_sym("K", cam_id))
            opt_intr = cal_vec[:4].copy()
            opt_dist = cal_vec[4:].copy()

            readout = 0.0
            readout_sigma = None
            if self._is_rs_camera(cam_id):
                readout = float(result_values.atVector(_sym("R", cam_id))[0])
                readout_sigma = readout_sigmas.get(cam_id)
                sig = (f" (sigma {readout_sigma * 1e3:.3f} ms)"
                       if readout_sigma is not None else "")
                logger.info(f"Camera {cam_id}: RS readout = {readout:.6f}s{sig}")

            logger.info(f"Camera {cam_id}: fx={opt_intr[0]:.2f} fy={opt_intr[1]:.2f} "
                         f"cx={opt_intr[2]:.2f} cy={opt_intr[3]:.2f} "
                         f"dist=[{', '.join(f'{d:.4f}' for d in opt_dist)}]")

            cameras.append(CameraParams(
                camera_id=cam_id, intrinsics=opt_intr, distortion=opt_dist,
                distortion_model=distortion_models[cam_id],
                image_size=image_sizes.get(cam_id, (0, 0)),
                rolling_shutter_readout=readout,
                rolling_shutter_readout_sigma=readout_sigma,
            ))

        # Extrinsics
        ext_results = []
        for cam_id in camera_ids:
            if cam_id == reference_camera or is_single:
                continue
            ek = _sym("E", cam_id)
            if not result_values.exists(ek):
                continue
            T_ext = result_values.atPose3(ek)
            R = T_ext.rotation().matrix()
            t_vec = T_ext.translation()
            rvec, _ = cv2.Rodrigues(R)
            ext_results.append(ExtrinsicTransform(
                from_camera=cam_id, to_camera=reference_camera,
                T_to_from=T_ext.matrix(), so3=rvec.flatten(),
                translation=np.array([t_vec[0], t_vec[1], t_vec[2]]),
            ))

        # Time offsets
        time_offsets = {}
        if use_dt:
            for cam_id in camera_ids:
                if cam_id == reference_camera:
                    continue
                dk = _sym("D", cam_id)
                if result_values.exists(dk):
                    dt = float(result_values.atVector(dk)[0])
                    time_offsets[cam_id] = dt
                    logger.info(f"Camera {cam_id}: time offset = {dt:.6f}s")

        # Board poses (ref camera trajectory) — with sanity check.
        # GTSAM LM can push weakly-constrained B variables to astronomical
        # translations when observations are sparse or dt is large relative
        # to frame rate. Detect and replace with identity to preserve
        # B-variable indexing (downstream code assumes position = B-index).
        # Z-depth filtering of diverged poses is handled in the pipeline
        # (between Stage 4 and Stage 5) where they can be properly removed
        # from the frame lists before graph rebuild.
        board_poses_out = []
        board_poses_gtsam = []
        n_sanitized = 0
        raw_norms = []
        raw_poses = []
        for i in range(bp_count):
            p = result_values.atPose3(_sym("B", i))
            raw_poses.append(p)
            raw_norms.append(float(np.linalg.norm(p.translation())))
        # Norm-only sanity: poses > 10x median distance are diverged.
        if raw_norms:
            med_dist = float(np.median(raw_norms))
            max_allowed = max(10.0 * med_dist, 20.0)  # at least 20m
        else:
            max_allowed = 20.0
        for i in range(bp_count):
            p = raw_poses[i]
            dist = raw_norms[i]
            if dist > max_allowed:
                board_poses_out.append(np.eye(4))
                board_poses_gtsam.append(gtsam.Pose3())
                n_sanitized += 1
            else:
                board_poses_out.append(p.matrix())
                board_poses_gtsam.append(p)
        if n_sanitized > 0:
            logger.warning(
                f"Board pose sanity: {n_sanitized}/{bp_count} poses exceeded "
                f"{max_allowed:.1f}m (median={med_dist:.3f}m) — replaced")

        # Reprojection errors — use the SHARED projection so residuals here
        # match exactly what _make_unified_factor minimizes, and what the
        # HTML report later shows.
        from mvis.calibration.factors.camera_projection import (
            project_observation, find_bracketing_poses, synced_time,
        )

        all_errors = []
        all_norms = []                    # pooled per-corner norms, all cameras
        # Board as fitted: refined landmarks when board heights are estimated,
        # the board's own positions otherwise, which the solve also held fixed. Per-corner mean
        # residual = the board-shape diagnostic.
        n_pts = self._board.num_points()
        pts_nominal = np.asarray(self._board.all_points_3d(), float)
        pts_eval = pts_nominal
        board_heights_out = None
        board_points_refined = None
        if self._options.board_heights:
            board_points_refined = np.array(
                [np.asarray(result_values.atPoint3(_sym("L", pid)), float)
                 for pid in range(n_pts)])
            board_heights_out = board_points_refined[:, 2] - pts_nominal[:, 2]
            pts_eval = board_points_refined
        bias_sum = np.zeros((n_pts, 2))
        bias_n = np.zeros(n_pts)
        for cam_id in camera_ids:
            dist_model = distortion_models[cam_id]
            cal_vec = result_values.atVector(_sym("K", cam_id))

            is_ref = (cam_id == reference_camera) or is_single
            T_ext = None
            if not is_ref:
                ek = _sym("E", cam_id)
                if result_values.exists(ek):
                    T_ext = result_values.atPose3(ek)

            # Optimized time offset and RS readout
            td_val = 0.0
            if use_dt and not is_ref:
                dk = _sym("D", cam_id)
                if result_values.exists(dk):
                    td_val = float(result_values.atVector(dk)[0])
            tr_val = 0.0
            if self._is_rs_camera(cam_id):
                rk = _sym("R", cam_id)
                if result_values.exists(rk):
                    tr_val = float(result_values.atVector(rk)[0])
            h_cam = image_sizes.get(cam_id, (640, 480))[1]

            errors = []
            frames = all_frames[cam_id]
            poses = all_board_poses[cam_id]
            for fi, (frame, pose) in enumerate(zip(frames, poses)):
                if pose is None:
                    continue
                t = synced_time(frame.timestamp, bp_timeline_t, snap_tol)
                bracket = find_bracketing_poses(t, bp_timeline_t, bp_timeline_idx)
                if bracket is None:
                    continue
                bp0_idx, bp1_idx, t0, t1 = bracket
                bp0 = result_values.atPose3(_sym("B", bp0_idx))
                bp1 = result_values.atPose3(_sym("B", bp1_idx))

                for obs in frame.observations:
                    proj = project_observation(
                        board_point=pts_eval[obs.point_id],
                        bp0=bp0, bp1=bp1,
                        time_meas=t, time_0=t0, time_1=t1,
                        cal_vec=cal_vec, distortion_model=dist_model,
                        T_ext=T_ext, td=td_val, tr=tr_val,
                        pixel_row=obs.row, image_height=h_cam,
                    )
                    if proj is not None:
                        errors.append(float(np.linalg.norm(proj - obs.pixel)))
                        bias_sum[obs.point_id] += proj - obs.pixel
                        bias_n[obs.point_id] += 1

            stats = self._errors_to_stats(errors, cam_id)
            if stats.num_observations > 0:
                all_errors.append(stats)
                all_norms.extend(errors)

        # Kalibr convention: RMS over the pooled corner norms of all cameras
        from mvis.calibration.reproj_stats import corner_stats
        overall_rms = corner_stats(all_norms)['rms'] if all_norms else 0.0

        # Map B-variable index to timestamp (parallel to board_poses_out).
        board_pose_times_out = [0.0] * bp_count
        for k, bi in enumerate(bp_timeline_idx):
            if 0 <= bi < bp_count:
                board_pose_times_out[bi] = float(bp_timeline_t[k])

        result = CalibrationResult(
            cameras=cameras, extrinsics=ext_results,
            reference_camera=reference_camera,
            board_poses=board_poses_out,
            board_poses_gtsam=board_poses_gtsam,
            board_pose_times=board_pose_times_out,
            valid_frame_indices=bp_source_frame_indices,
            time_offsets=time_offsets,
            board_points_refined=board_points_refined,
            board_heights=board_heights_out,
            board_height_n_obs=(lm_n_obs if board_heights_out is not None and lm_n_obs is not None
                                else None),
            board_point_bias=np.column_stack(
                [bias_sum / np.maximum(bias_n, 1)[:, None], bias_n]),
            factor_frames=factor_frames,
            snap_sync_tolerance_s=snap_tol,
            reprojection_errors=all_errors, overall_rms=overall_rms,
            diverged_pose_stats={
                "total": bp_count,
                "removed": n_sanitized,
                "kept": bp_count - n_sanitized,
            },
        )
        from mvis.calibration.corner_accounting import count_corners, guarded
        result.corner_accounting = guarded(
            count_corners, result, all_frames, self._board, image_sizes)
        return result

    # -- Reprojection helpers --

    def _errors_to_stats(self, errors: list[float], cam_id: int) -> ReprojectionErrorStats:
        # shared Kalibr-convention stats — same code path as the cam-IMU pipeline
        from mvis.calibration.reproj_stats import corner_stats, inlier_stats
        if not errors:
            return ReprojectionErrorStats(
                camera_id=cam_id, rms=0.0, mean=0.0, median=0.0,
                max=0.0, num_observations=0, p90=0.0, p95=0.0)
        s = corner_stats(errors)
        return ReprojectionErrorStats(
            camera_id=cam_id, rms=s['rms'], mean=s['mean'], median=s['median'],
            p90=s['p90'], p95=s['p95'], max=s['max'],
            num_observations=s['n_corners'], std=s['std'],
            inlier=inlier_stats(errors, getattr(getattr(self, '_options', None), 'pixel_sigma', 1.0)),
        )


def _board_height_gauge_factor(lm_keys, pts, sigma=1e-6):
    """Remove the plane the corner heights share with the board poses.

    Three residual rows on the heights h_i = L_i.z - z_i: the mean height,
    and its first moments about the board x and y axes (normalised by N and by
    the second moments, so each row is in metres). With these pinned to zero
    the board poses carry mean plane + tilt and the heights carry only the
    departure from that plane — the unique split.
    """
    pts = np.asarray(pts, float)
    z0 = pts[:, 2]
    w = board_height_gauge_weights(pts)

    def err(this, values, H):
        e = np.zeros(3)
        for i, k in enumerate(lm_keys):
            h = float(np.asarray(values.atPoint3(k))[2]) - z0[i]
            e += w[i] * h
            if H is not None:
                J = np.zeros((3, 3))
                J[:, 2] = w[i]
                H[i] = J
        return e
    return gtsam.CustomFactor(gtsam.noiseModel.Isotropic.Sigma(3, sigma),
                              gtsam.KeyVector(lm_keys), err)


# ======================================================================
# Unified projection factor
# ======================================================================

def _make_unified_factor(
    measured: np.ndarray,
    board_point: np.ndarray,
    pixel_row: float,
    image_height: float,
    time_meas: float,
    time_0: float,
    time_1: float,
    noise_model,
    bp0_key: int,
    bp1_key: int,
    lm_key: int,
    cal_key: int,
    ext_key: Optional[int] = None,
    td_key: Optional[int] = None,
    tr_key: Optional[int] = None,
    is_fisheye: bool = False,
    distortion_model: str = "",
):
    """Create a unified projection factor.

    Handles: pose interpolation (RS + temporal) + extrinsic + intrinsics.

    Keys (in order):
        bp0_key  — Pose3: board pose at t0
        bp1_key  — Pose3: board pose at t1
        lm_key   — Point3: board landmark
        cal_key  — Vector: intrinsics [fx,fy,cx,cy,d0,d1,...] (dim n_k >= 8)
        ext_key  — Pose3: extrinsic T_ref_cam (optional, for non-ref cameras)
        td_key   — Vector1: time offset (optional)
        tr_key   — Vector1: RS readout time (optional)

    Returns:
        (factor, error_func): CustomFactor and reference to prevent GC.
    """
    # Resolve distortion model (unregistered names raise here, not mid-solve)
    if distortion_model:
        _dist_model = distortion_model
    else:
        _dist_model = legacy_model(is_fisheye)
    check_model(_dist_model)
    meas = measured.copy()
    pt_3d = board_point.copy()
    row_f = float(pixel_row)
    h_f = max(float(image_height), 1.0)
    t_meas = float(time_meas)
    t0 = float(time_0)
    t1 = float(time_1)
    dt_interval = t1 - t0
    if abs(dt_interval) < 1e-12:
        dt_interval = 1.0

    # Build key list
    keys = [bp0_key, bp1_key, lm_key, cal_key]
    EXT_IDX = None
    TD_IDX = None
    TR_IDX = None
    if ext_key is not None:
        EXT_IDX = len(keys)
        keys.append(ext_key)
    if td_key is not None:
        TD_IDX = len(keys)
        keys.append(td_key)
    if tr_key is not None:
        TR_IDX = len(keys)
        keys.append(tr_key)

    # Delegate the forward projection to the shared camera_projection module
    # so the factor's residual and the HTML report's per-observation error
    # use the exact same math.
    from mvis.calibration.factors.camera_projection import project_observation

    def _project(bp0, bp1, point, cal_vec, T_ext=None, td_val=0.0, tr_val=0.0):
        return project_observation(
            board_point=point,
            bp0=bp0, bp1=bp1,
            time_meas=t_meas, time_0=t0, time_1=t1,
            cal_vec=cal_vec, distortion_model=_dist_model,
            T_ext=T_ext, td=td_val, tr=tr_val,
            pixel_row=row_f, image_height=h_f,
        )

    def error_func(this, values, H):
        bp0 = values.atPose3(bp0_key)
        bp1 = values.atPose3(bp1_key)
        point = values.atPoint3(lm_key)
        cal_vec = values.atVector(cal_key)
        T_ext = values.atPose3(ext_key) if ext_key is not None else None
        td_val = float(values.atVector(td_key)[0]) if td_key is not None else 0.0
        tr_val = float(values.atVector(tr_key)[0]) if tr_key is not None else 0.0

        projected = _project(bp0, bp1, point, cal_vec, T_ext, td_val, tr_val)

        if projected is None:
            res = np.array([1000.0, 1000.0])
            if H is not None:
                for i in range(len(keys)):
                    if i < 2:  # Pose3
                        H[i] = np.zeros((2, 6))
                    elif i == 2:  # Point3
                        H[i] = np.zeros((2, 3))
                    elif i == 3:  # Vector (cal), dim n_k
                        H[i] = np.zeros((2, len(cal_vec)))
                    elif EXT_IDX is not None and i == EXT_IDX:  # Pose3 ext
                        H[i] = np.zeros((2, 6))
                    else:  # Vector1
                        H[i] = np.zeros((2, 1))
            return res

        res = projected - meas

        if H is not None:
            from mvis.calibration.factors.pose_interpolation import (
                se3_interpolate_jacobians,
            )
            from mvis.calibration.factors.lie_groups import se3_adjoint

            # --- Forward pass with Jacobians ---

            # 1. SLERP: T_board and Jacobians w.r.t. B0, B1, λ
            t_corr = t_meas + td_val + (row_f / h_f) * tr_val
            lam = (t_corr - t0) / dt_interval
            T_board, J_slerp_B0, J_slerp_B1, v_slerp = \
                se3_interpolate_jacobians(bp0, bp1, lam)

            # 2. Extrinsic composition
            if T_ext is not None:
                T_cam = T_board.compose(T_ext)
                # d(T_board·E)/dT_board = Adj(E⁻¹)
                J_comp_self = se3_adjoint(T_ext.inverse())
                # d(T_board·E)/dE = I₆
            else:
                T_cam = T_board
                J_comp_self = np.eye(6)

            # 3. Project with analytic Jacobians, the same `_Projector` path for
            # every model (pose block w.r.t. the T_cam right tangent, as
            # gtsam PinholeCamera; point block = J_pc · Rᵀ)
            _zero_all_H = False
            try:
                _, J_proj_pose, J_proj_point, J_intr = make_projector(
                    _dist_model, cal_vec).project(T_cam, point, jac=True)
            except RuntimeError:
                _zero_all_H = True
            if not _zero_all_H:
                # J_intr is n_intr wide; the K variable may carry zero padding
                # (k_cal_vec: 4 + max(4, n_dist)) that the model never reads
                n = min(J_intr.shape[1], len(cal_vec))
                J_proj_cal = np.zeros((2, len(cal_vec)))
                J_proj_cal[:, :n] = J_intr[:, :n]

            if _zero_all_H:
                for i in range(len(keys)):
                    if i < 2:
                        H[i] = np.zeros((2, 6))
                    elif i == 2:
                        H[i] = np.zeros((2, 3))
                    elif i == 3:
                        H[i] = np.zeros((2, len(cal_vec)))
                    elif EXT_IDX is not None and i == EXT_IDX:
                        H[i] = np.zeros((2, 6))
                    else:
                        H[i] = np.zeros((2, 1))
                return res

            # --- Chain rule assembly ---

            # Intermediate: J_proj_pose through extrinsic composition
            J_pose_chain = J_proj_pose @ J_comp_self  # 2×6

            # H[0]: dr/dB₀ = J_proj_pose · J_comp_self · J_slerp_B0
            H[0] = J_pose_chain @ J_slerp_B0

            # H[1]: dr/dB₁ = J_proj_pose · J_comp_self · J_slerp_B1
            H[1] = J_pose_chain @ J_slerp_B1

            # H[2]: dr/dL — point Jacobian
            H[2] = J_proj_point

            # H[3]: dr/dK — intrinsics Jacobian
            H[3] = J_proj_cal

            # H[EXT]: dr/dE = J_proj_pose · I₆
            if EXT_IDX is not None:
                H[EXT_IDX] = J_proj_pose  # d(T_board·E)/dE = I₆

            # H[TD]: dr/dtd = J_pose_chain · v_slerp / Δt
            if TD_IDX is not None:
                H[TD_IDX] = (J_pose_chain @ v_slerp / dt_interval).reshape(2, 1)

            # H[TR]: dr/dtr = J_pose_chain · v_slerp · (row/H) / Δt
            if TR_IDX is not None:
                H[TR_IDX] = (J_pose_chain @ v_slerp * (row_f / h_f)
                             / dt_interval).reshape(2, 1)

        return res

    factor = gtsam.CustomFactor(noise_model, keys, error_func)
    return factor, error_func
