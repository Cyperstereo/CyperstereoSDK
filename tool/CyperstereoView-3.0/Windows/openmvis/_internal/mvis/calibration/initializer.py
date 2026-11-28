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
"""Step 1: Initialize calibration parameters.

Uses OpenCV for intrinsics (Zhang's method) and PnP for board poses.
Uses co-visible frame analysis for multi-camera extrinsics.
"""

import logging
import sys
from collections import defaultdict

import cv2
import numpy as np
import gtsam

from mvis.detection.board import Board
from mvis.io.frame_selector import nearest_ref
from mvis.core.camera_models import CUSTOM, check_model, default_dist
from mvis.core.types import FrameObservations

logger = logging.getLogger(__name__)


def _is_tag_grid(board) -> bool:
    """True for tag-grid boards (AprilTag AND ArUco): point_id = tag*4 + corner.
    I0 fix: the old check was hasattr(board, '_tag_size'), which ArUco grids
    fail (they carry _marker_size) — sending them down the chessboard row/col
    grouping and corrupting the vanishing-point focal init."""
    return ((hasattr(board, "_tag_size") or hasattr(board, "_marker_size"))
            and not _is_chessboard(board))


def _is_chessboard(board) -> bool:
    """Chessboard aliases _tag_size = square_size (detection/chessboard.py) for
    generic getattr() helpers, but its point_id = row * cols + col; without this
    exclusion _is_tag_grid() was True for it and both the VP init and the omni
    grid grouped chessboard corners as tag rows (garbage focal estimates)."""
    return getattr(board, "_tag_family", None) == "chessboard"


# ======================================================================
# Vanishing point focal length estimation (Kalibr approach)
# Based on: Hughes, Denny, Glavin, Jones, PAMI 2010
# and: Umbach & Jones, IEEE Trans. Instrumentation & Measurement, 2000
# ======================================================================

def _fit_circle(points: np.ndarray):
    """Fit a circle to 2D points using modified least squares.

    Args:
        points: Nx2 array of (x, y) coordinates. Need at least 3 points.

    Returns:
        (cx, cy, radius) or None if fitting fails.
    """
    n = len(points)
    if n < 3:
        return None

    x = points[:, 0]
    y = points[:, 1]

    sx = np.sum(x)
    sy = np.sum(y)
    sxx = np.sum(x * x)
    sxy = np.sum(x * y)
    syy = np.sum(y * y)
    sxxx = np.sum(x * x * x)
    sxxy = np.sum(x * x * y)
    sxyy = np.sum(x * y * y)
    syyy = np.sum(y * y * y)

    A = n * sxx - sx * sx
    B = n * sxy - sx * sy
    C = n * syy - sy * sy
    D = 0.5 * (n * sxyy - sx * syy + n * sxxx - sx * sxx)
    E = 0.5 * (n * sxxy - sy * sxx + n * syyy - sy * syy)

    denom = A * C - B * B
    if abs(denom) < 1e-12:
        return None

    cx = (D * C - B * E) / denom
    cy = (A * E - B * D) / denom
    radius = float(np.mean(np.sqrt((x - cx) ** 2 + (y - cy) ** 2)))

    if radius < 1e-6 or not np.isfinite(radius):
        return None

    return (cx, cy, radius)


def _intersect_circles(x1, y1, r1, x2, y2, r2):
    """Find intersection points of two circles.

    Returns:
        List of 0 or 2 intersection points as np.ndarray.
    """
    d = np.hypot(x1 - x2, y1 - y2)
    if d > r1 + r2 or d < abs(r1 - r2) or d < 1e-12:
        return []

    a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
    h_sq = r1 * r1 - a * a
    if h_sq < 0:
        return []
    h = np.sqrt(h_sq)

    x3 = x1 + a * (x2 - x1) / d
    y3 = y1 + a * (y2 - y1) / d

    if h < 1e-10:
        return []  # tangent — not useful for vanishing points

    return [
        np.array([x3 + h * (y2 - y1) / d, y3 - h * (x2 - x1) / d]),
        np.array([x3 - h * (y2 - y1) / d, y3 + h * (x2 - x1) / d]),
    ]


def _pnp_worker(input_path: str, output_path: str) -> None:
    """Worker function for multiprocessing PnP. Runs in a spawned process."""
    import pickle
    import cv2 as _cv2
    import numpy as _np

    _cv2.setNumThreads(1)

    with open(input_path, "rb") as f:
        data = pickle.load(f)

    board_pts = data["board_pts"]
    intrinsics = _np.array(data["intrinsics"])
    dist_coeffs = _np.array(data["dist_coeffs"])
    dm = check_model(data["distortion_model"])
    # same usable-ray floor as the in-process init_board_pose
    min_rays = max(int(data.get("min_points_for_pose", 4)), _MIN_PNP_RAYS)
    max_reprojection_px = float(data.get("max_reprojection_px", 0.0))
    _flag = (_cv2.SOLVEPNP_IPPE if data.get("pnp_method") == "ippe"
             else _cv2.SOLVEPNP_ITERATIVE)
    K = _np.array([
        [intrinsics[0], 0, intrinsics[2]],
        [0, intrinsics[1], intrinsics[3]],
        [0, 0, 1],
    ])

    results = []
    for obs_data in data["frame_data"]:
        if len(obs_data) < 4:
            results.append(None)
            continue
        obj_pts = _np.array([board_pts[pid] for pid, _ in obs_data], dtype=_np.float64)
        img_pts = _np.array([px for _, px in obs_data], dtype=_np.float64)
        try:
            if dm in _BEARING_MODELS:
                # the same solve and gates as the in-process path
                pose = board_pose_pnp(
                    obj_pts, img_pts, K, dist_coeffs, dm, _flag, min_rays, max_reprojection_px)
                ok = pose is not None
                if ok:
                    rvec, tvec, _ = pose
            elif dm == "radtan":
                try:
                    ok, rvec, tvec = _cv2.solvePnP(
                        obj_pts, img_pts, K, dist_coeffs,
                        flags=_flag)
                except _cv2.error:
                    ok, rvec, tvec = _cv2.solvePnP(
                        obj_pts, img_pts, K, dist_coeffs,
                        flags=_cv2.SOLVEPNP_ITERATIVE)
            else:
                raise ValueError(f"Unknown distortion model: {dm!r}")
            if ok:
                R, _ = _cv2.Rodrigues(rvec)
                if not (_np.isfinite(R).all() and _np.isfinite(tvec).all()):
                    results.append(None)
                    continue
                T = _np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = tvec.flatten()
                results.append(_np.linalg.inv(T).tolist())
            else:
                results.append(None)
        except Exception:
            results.append(None)

    with open(output_path, "wb") as f:
        pickle.dump(results, f)


class Initializer:
    """Initialize calibration parameters for the optimizer."""

    def __init__(self, board: Board):
        self._board = board

    def init_intrinsics_opencv(
        self,
        frames: list[FrameObservations],
        image_size: tuple[int, int],
        distortion_model: str = "radtan",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Initialize camera intrinsics.

        For radtan: uses OpenCV calibrateCamera (Zhang's method) — reliable for pinhole.
        For equidistant: uses vanishing point focal length estimation (Kalibr/Hughes PAMI 2010)
          — robust to high fisheye distortion where OpenCV fisheye.calibrate is fragile.

        Args:
            frames: List of frame observations for a single camera.
            image_size: (width, height) of the images.
            distortion_model: a `core.camera_models.MODELS` name.

        Returns:
            (intrinsics, distortion): intrinsics=[fx,fy,cx,cy], distortion coefficients.
        """
        if len(frames) < 1:
            raise ValueError("Need at least 1 frame for initialization")

        # Radtan: use OpenCV calibrateCamera directly
        # All other models (equidistant, fov, double_sphere, eucm, ucm, omni_radtan;
        # kb4 via equidistant) use VP init
        if distortion_model == "radtan":
            return self._init_intrinsics_opencv_radtan(frames, image_size)
        if distortion_model == "kb4":      # the same polynomial, single solve
            intr, d4 = self.init_intrinsics_opencv(frames, image_size, "equidistant")
            return intr, np.asarray(d4, float)[:4]

        w, h = image_size
        cx = (w - 1.0) / 2.0
        cy = (h - 1.0) / 2.0

        # Estimate focal length from vanishing points.
        # Fit circles to parallel lines on the board (rows AND columns)
        # and intersect circle pairs to estimate the focal length.
        # Using both directions doubles the number of estimates and
        # makes the method more robust for chessboards with few rows.
        f_guesses = []

        board_cols = getattr(self._board, '_cols', None) or 1
        is_apriltag = _is_tag_grid(self._board)   # I0 fix (ArUco); not Chessboard

        for frame in frames:
            if frame.num_points < self._board.min_points_for_pose():
                continue

            # Group image points by corner row and corner column.
            #
            # For AprilTag: point_id = tag_id * 4 + corner_idx
            #   tag_id = point_id // 4
            #   tag_row = tag_id // board_cols, tag_col = tag_id % board_cols
            #   corner_idx: 0=BL, 1=BR, 2=TR, 3=TL
            #   corner_row = tag_row * 2 + (1 if corner in {2,3} else 0)
            #   corner_col = tag_col * 2 + (1 if corner in {1,2} else 0)
            #
            # For chessboard: corner_row = point_id // board_cols
            #                  corner_col = point_id % board_cols
            rows_pts = {}  # corner_row -> list of (u, v)
            cols_pts = {}  # corner_col -> list of (u, v)
            for obs in frame.observations:
                if is_apriltag:
                    tag_id = obs.point_id // 4
                    corner_idx = obs.point_id % 4
                    tag_row = tag_id // board_cols
                    tag_col = tag_id % board_cols
                    corner_row = tag_row * 2 + (1 if corner_idx >= 2 else 0)
                    corner_col = tag_col * 2 + (1 if corner_idx in (1, 2) else 0)
                else:
                    corner_row = obs.point_id // board_cols
                    corner_col = obs.point_id % board_cols
                rows_pts.setdefault(corner_row, []).append(obs.pixel)
                cols_pts.setdefault(corner_col, []).append(obs.pixel)

            # Collect f estimates from both row-wise and column-wise circles
            for line_pts in (rows_pts, cols_pts):
                valid = {k: pts for k, pts in line_pts.items()
                         if len(pts) >= 3}
                if len(valid) < 2:
                    continue

                circles = {}
                for idx, pts in valid.items():
                    result = _fit_circle(np.array(pts))
                    if result is not None:
                        circles[idx] = result

                if len(circles) < 2:
                    continue

                indices = sorted(circles.keys())
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        c1 = circles[indices[i]]
                        c2 = circles[indices[j]]
                        ipts = _intersect_circles(c1[0], c1[1], c1[2],
                                                   c2[0], c2[1], c2[2])
                        if len(ipts) == 2:
                            f_guess = np.linalg.norm(
                                ipts[0] - ipts[1]) / np.pi
                            if np.isfinite(f_guess) and f_guess > 10:
                                f_guesses.append(f_guess)

        if f_guesses:
            f0 = float(np.median(f_guesses))
            logger.info(f"Vanishing point init: f={f0:.2f} from {len(f_guesses)} estimates")
        else:
            f0 = None

        # Model-specific distortion defaults (core.camera_models), zero-padded
        # to >= 4: fov [omega], eucm [alpha, beta], ucm [alpha], omni [xi=1 per
        # Kalibr/camodocal, ...], double_sphere [xi, alpha] = Usenko 2018 /
        # Basalt [-0.2, 0.6]. NB: DS xi=0.0 sits on the boundary of the
        # negative-xi basin where typical fisheye DS solutions live; starting
        # there let the optimizer drift to a wrong (positive-xi) basin where fx
        # had to inflate ~2x to match observations. Found 2026-05-01.
        _d0 = default_dist(distortion_model)
        dist_coeffs = np.array(_d0 + [0.0] * max(0, 4 - len(_d0)))

        # Omni-radtan initialization (Kalibr/camodocal strategy):
        # 1. SVD line-fitting for focal length at each xi candidate
        # 2. Back-project PnP for pose estimation (not raw-pixel PnP)
        # 3. Pick xi with lowest reprojection error
        if distortion_model == "omni_radtan":
            from mvis.calibration.factors.custom_projections import (
                project_omni_radtan, unproject_omni_radtan)
            import cv2

            best_xi = 1.0
            best_fx = float(max(w, h))
            best_err = float('inf')

            for xi_try in [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5]:
                # --- SVD line-fitting for focal length (Kalibr method) ---
                # For each board row, fit a conic and extract focal length
                f_guesses_xi = []
                board_cols = getattr(self._board, '_cols', None) or 1
                is_apriltag = _is_tag_grid(self._board)   # I0 fix (ArUco); not Chessboard

                n_focal = min(20, len(frames))
                focal_indices = np.linspace(0, len(frames) - 1, n_focal, dtype=int)
                for fi in focal_indices:
                    frame = frames[fi]
                    if frame.num_points < 8:
                        continue
                    # Group by corner row
                    rows_pts = {}
                    for obs in frame.observations:
                        if is_apriltag:
                            tag_id = obs.point_id // 4
                            corner_idx = obs.point_id % 4
                            tag_row = tag_id // board_cols
                            corner_row = tag_row * 2 + (1 if corner_idx >= 2 else 0)
                        else:
                            corner_row = obs.point_id // board_cols
                        rows_pts.setdefault(corner_row, []).append(obs.pixel)

                    for row_idx, pts in rows_pts.items():
                        if len(pts) < 4:
                            continue
                        pts_arr = np.array(pts)
                        u = pts_arr[:, 0] - cx
                        v = pts_arr[:, 1] - cy
                        # Build P matrix: [u, v, 0.5, -0.5*(u²+v²)]
                        P = np.column_stack([u, v,
                                             np.full(len(u), 0.5),
                                             -0.5 * (u * u + v * v)])
                        # SVD null space
                        try:
                            _, _, Vt = np.linalg.svd(P)
                            C = Vt[-1]  # null space
                        except Exception:
                            continue
                        t = C[0]**2 + C[1]**2 + C[2] * C[3]
                        if t <= 0:
                            continue
                        d_val = 1.0 / np.sqrt(t)
                        nx = C[0] * d_val
                        ny = C[1] * d_val
                        # Skip radial lines
                        if np.sqrt(nx**2 + ny**2) > 0.95:
                            continue
                        nz = np.sqrt(max(0, 1.0 - nx**2 - ny**2))
                        if nz < 1e-6:
                            continue
                        gamma = abs(C[2] * d_val / nz)
                        if 10 < gamma < 10000:
                            f_guesses_xi.append(gamma)

                # SVD gives same fx for all xi — scale by xi to differentiate
                if not f_guesses_xi:
                    fx_base = f0 if f0 and f0 > 10 else float(max(w, h)) * 0.35
                else:
                    fx_base = float(np.median(f_guesses_xi))
                # Scale focal length for this xi (approximate relationship)
                fx_try = fx_base * (1.0 + xi_try) / 2.0

                cal_try = np.array([fx_try, fx_try, cx, cy, xi_try, 0, 0, 0, 0])   # p2 = 0

                # --- Back-project PnP for pose estimation ---
                # Kalibr trick: back-project pixels → 3D rays → normalized pinhole → PnP
                # Only reproject the same points used in PnP (index-aligned)
                total_err = 0.0
                n_pts = 0
                n_pnp = min(20, len(frames))
                pnp_indices = np.linspace(0, len(frames) - 1, n_pnp, dtype=int)
                for pi in pnp_indices:
                    frame = frames[pi]
                    if frame.num_points < 6:
                        continue
                    pts_3d = []
                    pts_norm = []
                    valid_obs = []  # track which obs were used
                    for obs in frame.observations:
                        ray = unproject_omni_radtan(obs.pixel, cal_try)
                        if ray is None or ray[2] <= 0.1:
                            continue
                        nx_val = ray[0] / ray[2]
                        ny_val = ray[1] / ray[2]
                        if abs(nx_val) > 5.0 or abs(ny_val) > 5.0:
                            continue
                        pts_norm.append([nx_val, ny_val])
                        pts_3d.append(self._board.point_3d(obs.point_id))
                        valid_obs.append(obs)

                    if len(pts_3d) < 4:
                        continue

                    pts_3d_np = np.array(pts_3d, dtype=np.float64)
                    pts_norm_np = np.array(pts_norm, dtype=np.float32)

                    try:
                        ok, rvec, tvec = cv2.solvePnP(
                            pts_3d_np, pts_norm_np, np.eye(3), None)
                        if not ok:
                            continue
                        import gtsam
                        R_mat, _ = cv2.Rodrigues(rvec)
                        T = np.eye(4)
                        T[:3, :3] = R_mat
                        T[:3, 3] = tvec.flatten()
                        pose = gtsam.Pose3(T)

                        # Reproject only the observations used in PnP
                        for obs, pt_3d in zip(valid_obs, pts_3d):
                            # solvePnP's [R|t] is T_cam_board: board -> cam is
                            # transformFrom. (transformTo, used until #26, scored
                            # every xi on inverted poses, ~250 px at every xi.)
                            # A wide lens can still pick a low xi here (a 180-deg lens:
                            # xi = 1 for a 2.2 fit); omni_schedule's P1 moves
                            # it, in the pipeline and in `mvis cam` alike.
                            p_cam = pose.transformFrom(gtsam.Point3(*pt_3d))
                            px, _, _ = project_omni_radtan(
                                np.asarray(p_cam, dtype=np.float64),
                                cal_try)
                            if px is not None:
                                total_err += np.linalg.norm(px - obs.pixel)
                                n_pts += 1
                    except Exception:
                        continue

                if n_pts > 0:
                    mean_err = total_err / n_pts
                    logger.info(f"  Omni grid: xi={xi_try:.1f} fx={fx_try:.1f} "
                                 f"err={mean_err:.1f}px ({n_pts} pts)")
                    if mean_err < best_err:
                        best_err = mean_err
                        best_xi = xi_try
                        best_fx = fx_try
                else:
                    logger.info(f"  Omni grid: xi={xi_try:.1f} fx={fx_try:.1f} "
                                 f"— no valid points")

            logger.info(f"Omni init: best xi={best_xi:.1f} fx={best_fx:.2f} "
                         f"(mean err={best_err:.1f}px)")
            intrinsics = np.array([best_fx, best_fx, cx, cy])
            dist_coeffs = np.array([best_xi, 0, 0, 0, 0])   # xi, k1, k2, p1, p2
            return intrinsics, dist_coeffs

        # Sanity check: focal length should be reasonable.
        # Even ultra-wide 220° fisheye lenses have f > 0.1 * image_width.
        if f0 is not None and f0 > 0.1 * w and f0 < 5.0 * w:
            intrinsics = np.array([f0, f0, cx, cy])
            logger.info(f"VP init ({distortion_model}): fx={f0:.2f} fy={f0:.2f} "
                         f"cx={cx:.2f} cy={cy:.2f} dist={dist_coeffs}")
            return intrinsics, dist_coeffs

        # Heuristic fallback
        f0 = w * 0.35
        logger.warning(f"VP init failed for {distortion_model}, using heuristic f={f0:.2f}")
        intrinsics = np.array([f0, f0, cx, cy])
        logger.info(f"Init ({distortion_model}): fx={f0:.2f} fy={f0:.2f} "
                     f"cx={cx:.2f} cy={cy:.2f} dist={dist_coeffs}")
        return intrinsics, dist_coeffs

    def _init_intrinsics_opencv_radtan(
        self,
        frames: list[FrameObservations],
        image_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Initialize radtan intrinsics using OpenCV calibrateCamera (Zhang's method)."""
        obj_points_list, img_points_list = [], []
        for frame in frames:
            if frame.num_points < self._board.min_points_for_pose():
                continue
            obj_pts = np.array(
                [self._board.point_3d(obs.point_id) for obs in frame.observations],
                dtype=np.float32)
            img_pts = np.array(
                [obs.pixel for obs in frame.observations], dtype=np.float32)
            obj_points_list.append(obj_pts)
            img_points_list.append(img_pts)

        if len(obj_points_list) < 3:
            raise ValueError(
                f"Need at least 3 frames for radtan init, got {len(obj_points_list)}")

        w, h = image_size
        rms, K, dist, _, _ = cv2.calibrateCamera(
            obj_points_list, img_points_list, (w, h), None, None)
        intrinsics = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
        d = dist.flatten()
        dist_coeffs = np.array([
            d[0] if len(d) > 0 else 0.0, d[1] if len(d) > 1 else 0.0,
            d[2] if len(d) > 2 else 0.0, d[3] if len(d) > 3 else 0.0])

        logger.info(f"OpenCV calibrateCamera RMS: {rms:.4f}")
        logger.info(f"Init (radtan): fx={intrinsics[0]:.2f} fy={intrinsics[1]:.2f} "
                     f"cx={intrinsics[2]:.2f} cy={intrinsics[3]:.2f} dist={dist_coeffs}")
        return intrinsics, dist_coeffs

    def init_board_pose(
        self,
        frame: FrameObservations,
        K: np.ndarray,
        dist_coeffs: np.ndarray,
        distortion_model: str = "radtan",
        max_reprojection_px: float = 0.0,
        pnp_method: str = "iterative",
    ) -> gtsam.Pose3 | None:
        """Estimate board pose for a single frame using PnP.

        Args:
            frame: Frame observations.
            K: 3x3 camera matrix.
            dist_coeffs: Distortion coefficients.
            distortion_model: a `core.camera_models` name (unknown -> raise).

        Returns:
            gtsam.Pose3 or None if PnP fails.
        """
        check_model(distortion_model)
        if frame.num_points < self._board.min_points_for_pose():
            return None

        obj_pts = np.array(
            [self._board.point_3d(obs.point_id) for obs in frame.observations],
            dtype=np.float64,
        )
        img_pts = np.array(
            [obs.pixel for obs in frame.observations], dtype=np.float64,
        )

        # Use RANSAC-based PnP to handle outlier tag detections.
        # pnp_method "ippe" (planar boards): SOLVEPNP_IPPE resolves the planar
        # two-fold ambiguity that makes ITERATIVE flip pose basins — the MOC
        # fix (cam_imu_init), extended to the cam-only path for T5 (Reverb G2,
        # 640x480 + 3 cm tags) where the flip cluster poisoned the pair solve.
        # A cv2.error under IPPE falls through to the ITERATIVE fallback below.
        pnp_flag = (cv2.SOLVEPNP_IPPE if pnp_method == "ippe"
                    else cv2.SOLVEPNP_ITERATIVE)

        if distortion_model in _BEARING_MODELS:
            # Wide-FOV models: PnP on bearing rays, gated in pixels on the chosen pose.
            # >= 2 tags' worth of usable rays: with fewer (steep edge-of-FOV views) the
            # planar solve returns confident but wrong poses
            pose = board_pose_pnp(
                obj_pts, img_pts, K, dist_coeffs, distortion_model, pnp_flag,
                max(self._board.min_points_for_pose(), _MIN_PNP_RAYS), max_reprojection_px)
            if pose is None:
                return None
            rvec, tvec, _ = pose
        elif distortion_model == "radtan":
            ransac_thresh = 8.0  # px, RANSAC inlier threshold
            try:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj_pts, img_pts, K, dist_coeffs,
                    reprojectionError=ransac_thresh,
                    iterationsCount=100,
                    flags=pnp_flag,
                )
            except cv2.error:
                # Fall back to non-RANSAC iterative on failure
                try:
                    success, rvec, tvec = cv2.solvePnP(
                        obj_pts, img_pts, K, dist_coeffs,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                    )
                    inliers = None
                except cv2.error:
                    return None

            if not success:
                return None

            # Reprojection error check on RANSAC inliers only
            if max_reprojection_px > 0:
                try:
                    reproj, _ = cv2.projectPoints(
                        obj_pts, rvec, tvec, K, dist_coeffs)
                    reproj = reproj.reshape(-1, 2)
                    errs = np.linalg.norm(img_pts - reproj, axis=1)
                    # Use inlier errors only (RANSAC already filtered outliers)
                    if inliers is not None and len(inliers) > 0:
                        inlier_idx = inliers.flatten()
                        errs = errs[inlier_idx]
                    median_err = float(np.median(errs))
                    if median_err > max_reprojection_px:
                        return None
                except cv2.error:
                    pass  # can't check — proceed anyway
        else:       # unreachable after check_model; never treat as radtan
            raise ValueError(f"Unknown distortion model: {distortion_model!r}")

        R, _ = cv2.Rodrigues(rvec)
        if not (np.isfinite(R).all() and np.isfinite(tvec).all()):
            # Degenerate PnP yields NaN — gtsam.Rot3 segfaults on non-finite input, so reject the frame.
            return None
        rot3 = gtsam.Rot3(R)
        t = tvec.flatten()
        T_cam_board = gtsam.Pose3(rot3, gtsam.Point3(t[0], t[1], t[2]))
        # GTSAM GenericProjectionFactor expects T_world_cam = T_board_cam
        return T_cam_board.inverse()

    def init_board_poses(
        self,
        frames: list[FrameObservations],
        intrinsics: np.ndarray,
        dist_coeffs: np.ndarray,
        distortion_model: str = "radtan",
        use_subprocess: bool = False,
        max_reprojection_px: float = 0.0,
        pnp_method: str = "iterative",
    ) -> list[gtsam.Pose3 | None]:
        """Initialize board poses for all frames.

        Args:
            frames: List of frame observations.
            intrinsics: [fx, fy, cx, cy].
            dist_coeffs: Distortion coefficients.
            distortion_model: "radtan" or "equidistant".
            use_subprocess: If True, run PnP in a child process to survive segfaults.
            max_reprojection_px: If >0, discard poses with median reprojection
                error above this threshold.

        Returns:
            List of Pose3 (or None for failed frames).
        """
        if use_subprocess:
            return self._init_board_poses_subprocess(
                frames, intrinsics, dist_coeffs, distortion_model,
                pnp_method=pnp_method, max_reprojection_px=max_reprojection_px)

        K = np.array([
            [intrinsics[0], 0, intrinsics[2]],
            [0, intrinsics[1], intrinsics[3]],
            [0, 0, 1],
        ])

        poses = []
        for frame in frames:
            pose = self.init_board_pose(
                frame, K, dist_coeffs, distortion_model,
                max_reprojection_px=max_reprojection_px,
                pnp_method=pnp_method)
            poses.append(pose)

        n_success = sum(1 for p in poses if p is not None)
        logger.info(f"Board pose init: {n_success}/{len(frames)} frames succeeded")
        return poses

    def _init_board_poses_subprocess(
        self,
        frames: list[FrameObservations],
        intrinsics: np.ndarray,
        dist_coeffs: np.ndarray,
        distortion_model: str,
        pnp_method: str = "iterative",
        max_reprojection_px: float = 0.0,
    ) -> list[gtsam.Pose3 | None]:
        """Run PnP in a child process to survive segfaults from OpenCV/GTSAM conflicts.

        Uses multiprocessing with 'spawn' start method to avoid fork() issues
        with loaded shared libraries (GTSAM, OpenCV).
        """
        import multiprocessing as mp
        import pickle
        import shutil
        import tempfile
        import os

        logger.info(f"Subprocess PnP for {len(frames)} frames...")

        # Serialize data for the child process
        board_pts = {pid: self._board.point_3d(pid).tolist()
                     for pid in range(self._board.num_points())}

        frame_data = []
        for frame in frames:
            obs_data = [(o.point_id, o.pixel.tolist()) for o in frame.observations]
            frame_data.append(obs_data)

        # Both pickles live in a private (0700) directory of our own, so no
        # other local user can pre-place the file the parent unpickles.
        tmp_dir = tempfile.mkdtemp(prefix="mvis_pnp_")
        try:
            input_path = os.path.join(tmp_dir, "pnp_in.pkl")
            output_path = os.path.join(tmp_dir, "pnp_out.pkl")
            with open(input_path, "wb") as f:
                pickle.dump({
                    "board_pts": board_pts,
                    "frame_data": frame_data,
                    "intrinsics": intrinsics.tolist(),
                    "dist_coeffs": dist_coeffs.tolist(),
                    "distortion_model": check_model(distortion_model),
                    "pnp_method": pnp_method,
                    "min_points_for_pose": int(self._board.min_points_for_pose()),
                    "max_reprojection_px": float(max_reprojection_px),
                }, f)

            # Use 'spawn' to create a fresh process (no fork)
            ctx = mp.get_context('spawn')
            p = ctx.Process(target=_pnp_worker, args=(input_path, output_path))
            p.start()
            p.join(timeout=300)

            if p.exitcode != 0:
                logger.error(f"PnP subprocess failed (exit {p.exitcode})")
                if p.is_alive():        # join timeout: reap it before the dir goes
                    p.kill()
                    p.join()
                return [None] * len(frames)

            with open(output_path, "rb") as f:
                pose_matrices = pickle.load(f)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        # Convert to gtsam.Pose3
        poses = []
        for mat in pose_matrices:
            if mat is not None:
                poses.append(gtsam.Pose3(np.array(mat)))
            else:
                poses.append(None)

        n_success = sum(1 for p in poses if p is not None)
        logger.info(f"Board pose init (subprocess): {n_success}/{len(frames)} frames succeeded")
        return poses

    def init_extrinsics(
        self,
        all_frames: dict[int, list[FrameObservations]],
        all_poses: dict[int, list[gtsam.Pose3 | None]],
        reference_camera: int = 0,
    ) -> dict[int, gtsam.Pose3]:
        """Initialize extrinsics from co-visible board observations.

        For each non-reference camera i, find frames where both camera i and
        the reference camera observe the board. Compute:
            T_ref_i = T_ref_board * T_i_board^{-1}

        Args:
            all_frames: camera_id → list of FrameObservations.
            all_poses: camera_id → list of Pose3 (from init_board_poses).
            reference_camera: Reference camera ID.

        Returns:
            camera_id → Pose3 (transform from camera i to reference camera).
        """
        extrinsics = {}

        # Build timestamp→pose lookup for reference camera
        ref_frames = all_frames.get(reference_camera, [])
        ref_poses = all_poses.get(reference_camera, [])
        ref_by_time = {}
        for frame, pose in zip(ref_frames, ref_poses):
            if pose is not None:
                ref_by_time[frame.timestamp] = pose
        # Reference frame period from every reference frame, posed or not, so
        # PnP failures do not widen the nearest-time pairing tolerance.
        ref_period = self._median_interval([f.timestamp for f in ref_frames])

        for cam_id in all_frames:
            if cam_id == reference_camera:
                continue

            cam_frames = all_frames[cam_id]
            cam_poses = all_poses[cam_id]
            estimates = []

            for frame, cam_pose in zip(cam_frames, cam_poses):
                if cam_pose is None:
                    continue
                ref_pose = ref_by_time.get(frame.timestamp)
                if ref_pose is None:
                    continue
                # Board poses are T_board_cam (GTSAM convention).
                # ref_pose = T_board_cam_ref, cam_pose = T_board_cam_i
                # We want T_cam_i_cam_ref = T_cam_i_board * T_board_cam_ref
                #                         = cam_pose^{-1} * ref_pose
                T_cam_i_ref = cam_pose.inverse().compose(ref_pose)
                estimates.append(T_cam_i_ref)

            source = "co-visible frames"
            if len(estimates) < self.MIN_EXACT_EXTRINSIC_PAIRS and ref_by_time:
                # Unsynchronised cameras (separate devices on a shared host
                # clock) rarely share an exact stamp; one or two accidental
                # collisions must not leave a 1-2 sample init. Pair each frame
                # with the nearest reference frame within half a reference
                # period (exact matches pair with gap 0) and keep the larger set.
                nearest = self._nearest_time_estimates(
                    cam_frames, cam_poses, ref_by_time, cam_id, reference_camera,
                    n_exact=len(estimates), ref_period=ref_period)
                if len(nearest) > len(estimates):
                    estimates = nearest
                    source = "nearest-time pairs"

            if not estimates:
                logger.warning(
                    f"No co-visible frames for camera {cam_id} and reference {reference_camera}. "
                    "Using identity extrinsic."
                )
                extrinsics[cam_id] = gtsam.Pose3()
                continue

            # Average using the first estimate as reference and composing deltas
            avg_pose = self._average_poses(estimates)
            extrinsics[cam_id] = avg_pose
            logger.info(
                f"Camera {cam_id} extrinsic init from {len(estimates)} {source}"
            )

        return extrinsics

    def init_extrinsics_trajectory(
        self,
        all_frames: dict[int, list[FrameObservations]],
        all_poses: dict[int, list[gtsam.Pose3 | None]],
        reference_camera: int = 0,
        image_sizes: dict[int, tuple[int, int]] | None = None,
        skip_temporal: bool = False,
    ) -> tuple[dict[int, gtsam.Pose3], dict[int, float], dict[int, set[float]]]:
        """Initialize extrinsics and time offsets via trajectory alignment (Stage 1.5).

        Uses angular velocity cross-correlation for temporal alignment,
        RANSAC hand-eye calibration for rotation, RANSAC for translation.
        Returns raw outlier-frame timestamps from RANSAC for downstream
        downweighting.  Timestamps remain stable when later stages filter or
        reorder their frame lists; integer list positions do not.

        Args:
            all_frames: camera_id → list of FrameObservations.
            all_poses: camera_id → list of Pose3 (from init_board_poses).
            reference_camera: Reference camera ID.

        Returns:
            (extrinsics, time_offsets, outlier_frame_timestamps):
                extrinsics: camera_id → Pose3 (T_cam_i_ref)
                time_offsets: camera_id → dt (seconds)
                outlier_frame_timestamps: camera_id → raw timestamps of frames
                    with unreliable tracking
        """
        from mvis.calibration.pose_alignment import align_cameras

        # Build trajectories: filter out None poses
        trajectories = {}
        timestamps = {}
        for cam_id in all_frames:
            poses = []
            ts = []
            for frame, pose in zip(all_frames[cam_id], all_poses[cam_id]):
                if pose is not None:
                    poses.append(pose)
                    ts.append(frame.timestamp)
            if poses:
                trajectories[cam_id] = poses
                timestamps[cam_id] = ts

        results = align_cameras(trajectories, timestamps, reference_camera,
                                 all_frames=all_frames,
                                 image_sizes=image_sizes,
                                 skip_temporal=skip_temporal)

        extrinsics = {}
        time_offsets = {}
        all_outlier_frame_timestamps = {}
        for cam_id, (dt, T_cam_ref, outlier_frame_timestamps) in results.items():
            extrinsics[cam_id] = T_cam_ref
            time_offsets[cam_id] = dt
            for outlier_cam_id, timestamps in outlier_frame_timestamps.items():
                all_outlier_frame_timestamps.setdefault(
                    outlier_cam_id, set()).update(timestamps)
            t = T_cam_ref.translation()
            logger.info(
                f"Camera {cam_id} trajectory alignment: dt={dt:.6f}s, "
                f"baseline={np.linalg.norm([t[0], t[1], t[2]]):.4f}m"
            )

        return extrinsics, time_offsets, all_outlier_frame_timestamps

    # Below this many exact-stamp pairs, init_extrinsics also tries
    # nearest-time pairing (an accidental stamp collision is not a sync).
    MIN_EXACT_EXTRINSIC_PAIRS = 3

    @staticmethod
    def _median_interval(stamps) -> float | None:
        """Median interval between distinct sorted stamps (None if < 2)."""
        t = np.unique(np.asarray(list(stamps), dtype=float))
        if len(t) < 2:
            return None
        return float(np.median(np.diff(t)))

    @staticmethod
    def _nearest_time_estimates(cam_frames, cam_poses, ref_by_time, cam_id,
                                reference_camera, n_exact: int = 0,
                                ref_period: float | None = None,
                                ) -> list[gtsam.Pose3]:
        """T_cam_i_ref estimates from nearest-in-time reference frames.

        Pairs are accepted within half the reference frame period
        (``ref_period``, by default the median interval of the posed
        reference stamps). The board moves between the two stamps, so this is
        an initial guess only (the joint solve refines extrinsics and time
        offsets).

        Only the offset modulo one period is bounded: a clock offset of
        k + f periods with |f| < 0.5 pairs every frame with the reference
        frame k periods away, giving a non-identity init from mismatched
        boards. Identity results only when nothing lies within tolerance.
        """
        ref_t = np.array(sorted(ref_by_time))
        if ref_period is None:
            ref_period = Initializer._median_interval(ref_t)
        if ref_period is None or ref_period <= 0 or len(ref_t) == 0:
            return []
        tol = 0.5 * ref_period
        estimates, gaps = [], []
        for frame, cam_pose in zip(cam_frames, cam_poses):
            if cam_pose is None:
                continue
            k = int(nearest_ref(frame.timestamp, ref_t)[0])
            gap = abs(ref_t[k] - frame.timestamp)
            if gap > tol:
                continue
            estimates.append(cam_pose.inverse().compose(ref_by_time[ref_t[k]]))
            gaps.append(gap)
        if estimates:
            logger.info(
                f"Camera {cam_id}: {n_exact} exact co-visible stamps with "
                f"reference {reference_camera}; paired {len(estimates)} frames "
                f"by nearest time (median gap {1e3 * float(np.median(gaps)):.1f} ms, "
                f"tolerance {1e3 * tol:.1f} ms; stamps in s)")
        return estimates

    @staticmethod
    def _average_poses(poses: list[gtsam.Pose3]) -> gtsam.Pose3:
        """Average SE(3) poses using iterative tangent-space mean."""
        if len(poses) == 1:
            return poses[0]

        mean = poses[0]
        for _ in range(20):  # iterative refinement
            deltas = np.zeros(6)
            for p in poses:
                delta = mean.logmap(p)
                deltas += np.array([delta[i] for i in range(6)])
            deltas /= len(poses)
            if np.linalg.norm(deltas) < 1e-10:
                break
            mean = mean.expmap(deltas)

        return mean


# Rays further than this from the optical axis are dropped before the
# normalized-plane PnP: near 90 deg x/z, y/z blow up and a handful of such
# points dominates the solve (degenerate poses with depth ~1e14 m).
_MAX_PNP_RAY_DEG = 80.0
_MIN_PNP_RAYS = 8
_RANSAC_PX = 8.0        # RANSAC inlier threshold
# models whose PnP runs on bearing rays (cv2 only knows radtan coefficients)
_BEARING_MODELS = frozenset({"equidistant"}) | CUSTOM


def _full_cal_vec(K, dist_coeffs):
    """[fx, fy, cx, cy, dist...] with ALL of `dist_coeffs` (zero-padded to at
    least 4 — the custom kernels read up to cal_vec[7])."""
    d = list(np.asarray(dist_coeffs, float).ravel())
    return np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]] + d + [0.0] * max(0, 4 - len(d)))


def _bearing_points(img_pts, K, dist_coeffs, distortion_model):
    """Pixels -> normalized-plane points (x/z, y/z) for the wide-FOV models.

    Returns (points (M,2), keep mask (N,)). Pixels that fail to unproject or
    whose ray is more than _MAX_PNP_RAY_DEG off-axis are DROPPED; they are
    never passed on as raw pixel coordinates (which mixed units into the solve).
    """
    from mvis.camera.normalized_points import normalized_points
    # the model's own inverse (CameraModel), as every other init path; for equidistant not
    # cv2.fisheye: that returns a sentinel for valid edge pixels and can settle on a later
    # branch of the polynomial
    xy, keep = normalized_points(img_pts, distortion_model, _full_cal_vec(K, dist_coeffs),
                                 max_ray_deg=_MAX_PNP_RAY_DEG)
    return xy[keep].astype(np.float64), keep


def _model_project(p_cam, K, dist_coeffs, distortion_model):
    """Camera-frame points (N,3) -> pixels (N,2) with the true model; NaN where a point has no valid
    forward projection under that model."""
    from mvis.camera.camera_model import build_camera_model
    cam = build_camera_model(distortion_model, _full_cal_vec(K, dist_coeffs))
    return np.array([cam.project(p) for p in p_cam]).reshape(-1, 2)


def _best_planar_pose(obj_pts, img_pts, img_norm, cands, K, dist_coeffs, distortion_model):
    """Pick the board pose with the lowest median TRUE-model pixel error among `cands` [(rvec, tvec)]
    and both IPPE planar solutions (each refined with ITERATIVE on the normalized points). A
    candidate must be finite and have a valid forward projection under the model for >= 90% of the
    board; only those points are scored. Earlier candidates win ties. Returns (rvec, tvec, median
    pixel error), or (None, None, inf)."""
    cands = list(cands)
    try:
        _, rvs, tvs, _ = cv2.solvePnPGeneric(
            obj_pts, img_norm, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        rvs, tvs = [], []
    for r0, t0 in zip(rvs, tvs):
        try:
            _, r1, t1 = cv2.solvePnP(obj_pts, img_norm, np.eye(3), None,
                                     r0.copy(), t0.copy(), useExtrinsicGuess=True,
                                     flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:
            r1, t1 = r0, t0
        cands.append((r1, t1))
    best, best_err = (None, None), np.inf
    for r, t in cands:
        if not (np.isfinite(r).all() and np.isfinite(t).all()):
            continue
        R, _ = cv2.Rodrigues(r)
        p_cam = obj_pts @ R.T + t.reshape(1, 3)
        e = np.linalg.norm(_model_project(p_cam, K, dist_coeffs, distortion_model) - img_pts, axis=1)
        if np.isfinite(e).mean() < 0.9:
            continue
        med = float(np.nanmedian(e))
        if med < best_err:
            best, best_err = (r, t), med
    return (*best, best_err)


def board_pose_pnp(obj_pts, img_pts, K, dist_coeffs, distortion_model, pnp_flag, min_rays,
                   max_reprojection_px):
    """Board pose (board -> camera) from pixels for a bearing model, the one solve behind the
    camera-only initializer (in-process and worker) and the cam-IMU `pnp_pose_track`: PnP on the
    model's rays with pixels that have no ray dropped together with their board points, RANSAC
    plus the planar IPPE candidates, the best by true-model pixel error, then the pixel gate
    (`max_reprojection_px` > 0) on that pose over all kept corners.
    Returns (rvec, tvec, corners used), or None."""
    img_norm, keep = _bearing_points(img_pts, K, dist_coeffs, distortion_model)
    if keep.sum() < min_rays:
        return None
    obj_pts, img_pts = obj_pts[keep], img_pts[keep]
    # the RANSAC threshold is in normalized units here, not pixels
    f_mean = 0.5 * (float(K[0, 0]) + float(K[1, 1]))
    thresh = _RANSAC_PX / max(f_mean, 1e-9)
    cands = []
    try:
        # RANSAC often fails on steep peripheral views; the planar IPPE candidates still recover
        # the pose, so always choose among them.
        ok, rv, tv, _ = cv2.solvePnPRansac(obj_pts, img_norm, np.eye(3), None, reprojectionError=thresh,
                                           iterationsCount=100, flags=pnp_flag)
        if ok:
            cands.append((rv, tv))
    except cv2.error:
        try:
            ok, rv, tv = cv2.solvePnP(obj_pts, img_norm, np.eye(3), None,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                cands.append((rv, tv))
        except cv2.error:
            pass
    rvec, tvec, median_px = _best_planar_pose(
        obj_pts, img_pts, img_norm, cands, K, dist_coeffs, distortion_model)
    if pnp_flag != cv2.SOLVEPNP_IPPE and not median_px <= _RANSAC_PX:
        # Nothing fits: ITERATIVE RANSAC can settle on the wrong planar branch and the full-set IPPE
        # candidates don't tolerate outliers, so try the robust planar solve. Only then: as a
        # candidate on every frame it changes poses that already fit, and moved fits by up to 0.29 px
        # (mvis_elp/stereo_rs_bag12). Non-planar points raise cv2.error here.
        try:
            ok, rv, tv, _ = cv2.solvePnPRansac(obj_pts, img_norm, np.eye(3), None, reprojectionError=thresh,
                                               iterationsCount=100, flags=cv2.SOLVEPNP_IPPE)
            if ok:
                rvec, tvec, median_px = _best_planar_pose(
                    obj_pts, img_pts, img_norm, [*cands, (rv, tv)], K, dist_coeffs, distortion_model)
        except cv2.error:
            pass
    if rvec is None or (max_reprojection_px > 0 and median_px > max_reprojection_px):
        return None
    return rvec, tvec, int(keep.sum())
