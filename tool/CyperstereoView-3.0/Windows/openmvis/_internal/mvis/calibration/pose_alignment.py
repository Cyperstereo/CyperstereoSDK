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
"""Inter-camera pose alignment (Stage 1.5).

Given per-camera board pose trajectories from Stage 1, align all cameras
to the reference camera by estimating:
  - Time offset dt (Step A: robust angular velocity cross-correlation)
  - Rotation R_cam_i_ref (Step B: RANSAC hand-eye calibration)
  - Translation t_cam_i_ref (Step C: RANSAC matched-frame estimation)

Includes PnP outlier detection to remove bad poses before alignment.
"""

import logging
import random
from typing import Optional

import numpy as np
import gtsam

logger = logging.getLogger(__name__)


# ============================================================
# PnP Outlier Detection
# ============================================================

# Board depth (m) in the camera frame a sane board pose has; outside it a pose diverged.
_BOARD_Z_RANGE = (0.05, 20.0)


def velocity_jump_inliers(v, thr) -> np.ndarray:
    """Inlier mask over n = len(v) + 1 poses from the n - 1 neighbour velocities `v`:
    a pose is out when the velocity to either neighbour exceeds `thr`. Each caller
    computes `v` with its own dt rule and picks its own threshold."""
    n = len(v) + 1
    ok = np.ones(n, bool)
    for i in range(n):
        if i > 0 and v[i - 1] > thr:
            ok[i] = False
        if i < n - 1 and v[i] > thr:
            ok[i] = False
    return ok


def detect_pnp_outliers(
    poses: list[gtsam.Pose3],
    timestamps: list[float],
    omega_multiplier: float = 3.0,
) -> list[bool]:
    """Detect PnP outlier poses using angular velocity check.

    A pose is an outlier if the angular velocity to either neighbor
    exceeds omega_multiplier * 95th percentile.

    Args:
        poses: List of board poses.
        timestamps: Corresponding timestamps.
        omega_multiplier: Multiplier for the 95th percentile threshold.

    Returns:
        List of bools: True = inlier, False = outlier.
    """
    n = len(poses)
    if n < 3:
        return [True] * n

    # Compute all inter-frame angular velocities
    omegas = []
    for i in range(n - 1):
        dt = timestamps[i + 1] - timestamps[i]
        if dt < 1e-10:
            omegas.append(0.0)
            continue
        R_rel = poses[i].rotation().between(poses[i + 1].rotation())
        angle = np.linalg.norm(gtsam.Rot3.Logmap(R_rel))
        omegas.append(angle / dt)

    omegas = np.array(omegas)
    if len(omegas) == 0:
        return [True] * n

    threshold = np.percentile(omegas, 95) * omega_multiplier
    # Minimum threshold to avoid flagging everything in slow sequences
    threshold = max(threshold, 1.0)  # at least 1 rad/s

    # out when the angular velocity to either neighbour exceeds the threshold
    is_inlier = [bool(k) for k in velocity_jump_inliers(omegas, threshold)]

    n_outliers = sum(1 for x in is_inlier if not x)
    if n_outliers > 0:
        logger.info(f"PnP outlier detection: {n_outliers}/{n} poses flagged "
                     f"(omega threshold={threshold:.2f} rad/s)")
    return is_inlier


def filter_trajectory(
    poses: list[gtsam.Pose3],
    timestamps: list[float],
    is_inlier: list[bool],
) -> tuple[list[gtsam.Pose3], list[float]]:
    """Filter trajectory to keep only inlier poses."""
    p = [poses[i] for i in range(len(poses)) if is_inlier[i]]
    t = [timestamps[i] for i in range(len(timestamps)) if is_inlier[i]]
    return p, t


# ============================================================
# Angular velocity and relative rotations
# ============================================================

def compute_angular_velocities(
    poses: list[gtsam.Pose3],
    timestamps: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute angular velocity magnitude from a pose trajectory."""
    times, omega = [], []
    for i in range(len(poses) - 1):
        dt = timestamps[i + 1] - timestamps[i]
        if dt < 1e-10:
            continue
        R_rel = poses[i].rotation().between(poses[i + 1].rotation())
        angle = np.linalg.norm(gtsam.Rot3.Logmap(R_rel))
        times.append((timestamps[i] + timestamps[i + 1]) / 2.0)
        omega.append(angle / dt)
    return np.array(times), np.array(omega)


def compute_relative_rotations(
    poses: list[gtsam.Pose3],
    timestamps: list[float],
) -> tuple[list[gtsam.Rot3], np.ndarray]:
    """Compute relative rotations between consecutive frames."""
    rots, times = [], []
    for i in range(len(poses) - 1):
        rots.append(poses[i].rotation().between(poses[i + 1].rotation()))
        times.append((timestamps[i] + timestamps[i + 1]) / 2.0)
    return rots, np.array(times)


# ============================================================
# Step A: Temporal alignment (robust)
# ============================================================

def _median_filter(signal: np.ndarray, size: int = 3) -> np.ndarray:
    from scipy.ndimage import median_filter
    return median_filter(signal, size=size)


def _cumulative_rotation(omega: np.ndarray, times: np.ndarray) -> np.ndarray:
    dt = np.diff(times, prepend=times[0])
    dt[0] = 0.0
    return np.cumsum(omega * dt)




def _upsample_angular_velocity(poses, timestamps, target_hz=100.0):
    """Upsample a pose trajectory via SLERP and compute angular velocity at target rate.

    This gives much better temporal resolution than computing angular velocity
    directly from sparse poses (e.g., 5Hz → 100Hz = 20x better resolution).

    Args:
        poses: List of gtsam.Pose3.
        timestamps: List of float timestamps.
        target_hz: Target sample rate for the upsampled trajectory.

    Returns:
        (times, omega): Arrays of timestamps and angular velocity norms at target_hz.
    """
    from mvis.calibration.factors.pose_interpolation import se3_interpolate as _se3_interpolate

    if len(poses) < 2:
        return np.array([]), np.array([])

    ts = np.array(timestamps)
    dt_target = 1.0 / target_hz

    # Build dense time grid
    t_dense = np.arange(ts[0], ts[-1], dt_target)
    if len(t_dense) < 3:
        return compute_angular_velocities(poses, timestamps)

    # Interpolate poses at dense timestamps
    dense_poses = []
    pose_idx = 0
    for t in t_dense:
        # Advance to the right bracket
        while pose_idx < len(ts) - 2 and ts[pose_idx + 1] < t:
            pose_idx += 1
        t0, t1 = ts[pose_idx], ts[min(pose_idx + 1, len(ts) - 1)]
        dt_interval = t1 - t0
        if dt_interval < 1e-12:
            dense_poses.append(poses[pose_idx])
        else:
            lam = (t - t0) / dt_interval
            lam = max(0.0, min(1.0, lam))
            dense_poses.append(_se3_interpolate(poses[pose_idx], poses[min(pose_idx + 1, len(poses) - 1)], lam))

    # Compute angular velocity from dense poses
    return compute_angular_velocities(dense_poses, t_dense.tolist())


def _xcorr_offset(s_ref, s_cam, dt_sample, max_expected_dt):
    """Time offset between two signals on one grid of step `dt_sample` (s), by normalised
    cross-correlation; the peak is searched within |lag| <= max_expected_dt.

    Zero-means and normalises `s_ref` and `s_cam` in place. Returns (dt_est, peak_corr);
    None when a signal is too weak to normalise (norm < 1e-10), (0.0, None) when no lag
    lies within the range. Each caller keeps its own correlation thresholds.
    """
    s_ref -= s_ref.mean()
    s_cam -= s_cam.mean()
    n_r = np.linalg.norm(s_ref)
    n_c = np.linalg.norm(s_cam)
    if n_r < 1e-10 or n_c < 1e-10:
        return None
    s_ref /= n_r
    s_cam /= n_c

    corr = np.correlate(s_ref, s_cam, mode="full")
    lags = np.arange(-(len(s_ref) - 1), len(s_ref)) * dt_sample

    valid_mask = np.abs(lags) <= max_expected_dt
    if not np.any(valid_mask):
        return 0.0, None
    corr_valid = corr.copy()
    corr_valid[~valid_mask] = -np.inf
    peak_idx = np.argmax(corr_valid)
    return float(lags[peak_idx]), float(corr[peak_idx])


def estimate_time_offset_rotation(
    poses_ref, timestamps_ref, poses_cam, timestamps_cam,
    max_expected_dt=1.0,
):
    """Estimate time offset via angular velocity norm cross-correlation.

    Upsamples both pose trajectories to 100Hz via SLERP interpolation,
    then cross-correlates the angular velocity norms.

    Returns:
        Estimated time offset dt such that t_cam_corrected = t_cam + dt.
    """
    # Upsample to 100Hz for fine temporal resolution
    t_ref, omega_ref = _upsample_angular_velocity(poses_ref, timestamps_ref, target_hz=100.0)
    t_cam, omega_cam = _upsample_angular_velocity(poses_cam, timestamps_cam, target_hz=100.0)

    if len(omega_ref) < 10 or len(omega_cam) < 10:
        logger.warning("Too few angular velocity samples for time offset estimation")
        return 0.0

    # Median filter to remove interpolation artifacts
    omega_ref_f = _median_filter(omega_ref, size=5)
    omega_cam_f = _median_filter(omega_cam, size=5)

    # Resample to common grid for cross-correlation.
    # Use 100Hz (10ms resolution) but cap grid to 50,000 points to avoid
    # O(n²) blowup in np.correlate for long recordings or fake timestamps.
    dt_sample = 0.01  # 100Hz = 10ms resolution
    t_start = max(t_ref[0], t_cam[0])
    t_end = min(t_ref[-1], t_cam[-1])
    if t_end - t_start < 1.0:
        logger.warning("Overlapping time window < 1s, cannot estimate time offset")
        return 0.0

    max_grid_points = 50000
    duration = t_end - t_start
    if duration / dt_sample > max_grid_points:
        dt_sample = duration / max_grid_points

    t_grid = np.arange(t_start, t_end, dt_sample)
    if len(t_grid) < 100:
        return 0.0

    # Interpolate angular velocities onto common grid
    s_ref = np.interp(t_grid, t_ref, omega_ref_f)
    s_cam = np.interp(t_grid, t_cam, omega_cam_f)

    # Cross-correlate, peak within max_expected_dt
    xc = _xcorr_offset(s_ref, s_cam, dt_sample, max_expected_dt)
    if xc is None:
        logger.warning("Angular velocity signal too weak for time offset estimation")
        return 0.0
    dt_est, peak_corr = xc
    if peak_corr is None:
        return 0.0

    # Quality check — low confidence but still return the estimate.
    # The caller cross-checks PnP against the homography method: if they agree
    # within 50ms, the PnP value is used (finer resolution). If PnP is
    # truly garbage, the homography method will disagree and override it.
    if peak_corr < 0.15:
        logger.warning(f"Time offset peak correlation too low: {peak_corr:.3f} "
                        f"(peak at dt={dt_est*1000:.1f}ms)")
        return 0.0
    if peak_corr < 0.3:
        logger.warning(f"Time offset peak correlation marginal: {peak_corr:.3f} "
                        f"(peak at dt={dt_est*1000:.1f}ms, will cross-check with homography)")

    logger.info(f"Time offset: dt={dt_est:.6f}s (corr={peak_corr:.3f}, "
                 f"resolution={dt_sample*1000:.0f}ms, upsampled to 100Hz)")
    return dt_est


# ============================================================
# Step A2: Homography-based time offset (no intrinsics)
# ============================================================

def _compute_rotation_signal_homography(frames, image_size):
    """Compute per-frame rotation magnitude from homography decomposition.

    For each pair of consecutive frames, computes the inter-frame
    homography from matched tag corners, then extracts the approximate
    rotation angle via ``R ≈ K⁻¹ H K`` with a rough K from image
    dimensions. No calibrated intrinsics needed.

    This is more robust than PnP-based angular velocity for cross-FOV
    cameras because:
    - No intrinsic dependency (uses image_size only)
    - Isolates rotation from translation (homography decomposition)
    - Same physical rotation produces same angle regardless of FOV

    Args:
        frames: list of FrameObservations (with .observations, .timestamp).
        image_size: (width, height) of the camera image.

    Returns:
        (times, omega): arrays of mid-timestamps and rotation rates (rad/s).
    """
    import cv2 as _cv2

    if len(frames) < 2:
        return np.array([]), np.array([])

    w, h = image_size
    K = np.array([[w / 2, 0, w / 2],
                   [0, h / 2, h / 2],
                   [0, 0, 1]], dtype=np.float64)
    K_inv = np.linalg.inv(K)

    times: list[float] = []
    angles: list[float] = []
    for i in range(1, len(frames)):
        dt = frames[i].timestamp - frames[i - 1].timestamp
        if dt <= 0:
            continue
        pts_prev = {o.point_id: o.pixel for o in frames[i - 1].observations}
        pts_curr = {o.point_id: o.pixel for o in frames[i].observations}
        common = sorted(set(pts_prev) & set(pts_curr))
        if len(common) < 8:
            continue
        src = np.array([pts_prev[p] for p in common], dtype=np.float64)
        dst = np.array([pts_curr[p] for p in common], dtype=np.float64)
        H, mask = _cv2.findHomography(src, dst, _cv2.RANSAC, 3.0)
        if H is None:
            continue
        # Approximate rotation: R ≈ K⁻¹ H K (valid when translation << distance)
        R_approx = K_inv @ H @ K
        # Project to nearest rotation via SVD
        U, S, Vt = np.linalg.svd(R_approx)
        R_clean = U @ Vt
        if np.linalg.det(R_clean) < 0:
            R_clean = -R_clean
        trace_val = np.clip((np.trace(R_clean) - 1.0) / 2.0, -1.0, 1.0)
        angle = float(np.arccos(trace_val))
        angles.append(angle / dt)
        times.append(0.5 * (frames[i].timestamp + frames[i - 1].timestamp))
    return np.array(times), np.array(angles)


def estimate_time_offset_homography(
    ref_frames, ref_image_size,
    cam_frames, cam_image_size,
    max_expected_dt: float = 1.0,
):
    """Estimate time offset via homography-based rotation cross-correlation.

    Replaces PnP-based angular velocity with a homography decomposition
    that needs only image dimensions (no calibrated intrinsics). Robust
    for cross-FOV camera pairs (e.g. fisheye + pinhole).

    Args:
        ref_frames: list of FrameObservations for reference camera.
        ref_image_size: (width, height) of reference camera.
        cam_frames: list of FrameObservations for non-ref camera.
        cam_image_size: (width, height) of non-ref camera.
        max_expected_dt: search range for time offset (seconds).

    Returns:
        Estimated time offset dt (seconds).
    """
    t_ref, omega_ref = _compute_rotation_signal_homography(
        ref_frames, ref_image_size)
    t_cam, omega_cam = _compute_rotation_signal_homography(
        cam_frames, cam_image_size)

    if len(omega_ref) < 10 or len(omega_cam) < 10:
        logger.warning("Too few rotation samples for homography-based "
                         "time offset estimation")
        return 0.0

    # Median filter
    omega_ref_f = _median_filter(omega_ref, size=5)
    omega_cam_f = _median_filter(omega_cam, size=5)

    # Resample to common grid for cross-correlation.
    # Use 100Hz (10ms resolution) but cap grid to 50,000 points to avoid
    # O(n²) blowup in np.correlate for long recordings.
    dt_sample = 0.01
    t_start = max(t_ref[0], t_cam[0])
    t_end = min(t_ref[-1], t_cam[-1])
    if t_end - t_start < 1.0:
        logger.warning("Overlapping time window < 1s for homography dt")
        return 0.0

    max_grid_points = 50000
    duration = t_end - t_start
    if duration / dt_sample > max_grid_points:
        dt_sample = duration / max_grid_points

    t_grid = np.arange(t_start, t_end, dt_sample)
    if len(t_grid) < 100:
        return 0.0

    s_ref = np.interp(t_grid, t_ref, omega_ref_f)
    s_cam = np.interp(t_grid, t_cam, omega_cam_f)

    xc = _xcorr_offset(s_ref, s_cam, dt_sample, max_expected_dt)
    if xc is None:
        logger.warning("Rotation signal too weak for homography dt")
        return 0.0
    dt_est, peak_corr = xc
    if peak_corr is None:
        return 0.0

    if peak_corr < 0.3:
        logger.warning(f"Homography dt peak correlation too low: {peak_corr:.3f}")
        return 0.0

    logger.info(f"Time offset (homography): dt={dt_est:.6f}s "
                 f"(corr={peak_corr:.3f}, resolution={dt_sample*1000:.0f}ms)")
    return dt_est


# ============================================================
# Step B: RANSAC Rotation alignment
# ============================================================

def _handeye_error(R, dr_r, dr_c):
    """Compute rotation error for AX=XB: ||Log(dR_cam^{-1} * R * dR_ref * R^T)||."""
    predicted = R.compose(dr_r).compose(R.inverse())
    residual = gtsam.Rot3.Logmap(dr_c.between(predicted))
    return np.linalg.norm([residual[0], residual[1], residual[2]])


def solve_rotation_handeye(dR_ref, dR_cam):
    """Solve AX=XB with iterative optimization on SO(3)."""
    if not dR_ref:
        return gtsam.Rot3()

    estimates = [dr_c.compose(dr_r.inverse()) for dr_r, dr_c in zip(dR_ref, dR_cam)]
    R = estimates[0]
    for _ in range(20):
        deltas = np.zeros(3)
        for R_est in estimates:
            d = gtsam.Rot3.Logmap(R.between(R_est))
            deltas += np.array([d[0], d[1], d[2]])
        deltas /= len(estimates)
        if np.linalg.norm(deltas) < 1e-12:
            break
        R = R.compose(gtsam.Rot3.Expmap(deltas))

    for _ in range(50):
        grad = np.zeros(3)
        for dr_r, dr_c in zip(dR_ref, dR_cam):
            predicted = R.compose(dr_r).compose(R.inverse())
            res = gtsam.Rot3.Logmap(dr_c.between(predicted))
            grad += np.array([res[0], res[1], res[2]])
        grad /= len(dR_ref)
        if np.linalg.norm(grad) < 1e-10:
            break
        R = R.compose(gtsam.Rot3.Expmap(grad * 0.5))
    return R


def solve_rotation_ransac(dR_ref, dR_cam, n_iter=200, threshold=0.1):
    """RANSAC rotation alignment from relative rotation pairs.

    Args:
        dR_ref: Relative rotations from ref camera.
        dR_cam: Corresponding relative rotations from other camera.
        n_iter: RANSAC iterations.
        threshold: Inlier threshold in radians (~6 degrees).

    Returns:
        (R_cam_ref, inlier_indices).
    """
    n = len(dR_ref)
    if n < 2:
        return gtsam.Rot3(), list(range(n))

    best_R = gtsam.Rot3()
    best_inliers = []

    rng = random.Random(0)  # seed RANSAC for run-to-run reproducibility
    for _ in range(n_iter):
        # Sample 2 pairs
        idx = rng.sample(range(n), min(2, n))
        dr_r_sample = [dR_ref[i] for i in idx]
        dr_c_sample = [dR_cam[i] for i in idx]

        R_est = solve_rotation_handeye(dr_r_sample, dr_c_sample)

        # Count inliers
        inliers = []
        for k in range(n):
            err = _handeye_error(R_est, dR_ref[k], dR_cam[k])
            if err < threshold:
                inliers.append(k)

        if len(inliers) > len(best_inliers):
            best_R = R_est
            best_inliers = inliers

    # Re-estimate from inliers
    if len(best_inliers) >= 2:
        dr_r_in = [dR_ref[i] for i in best_inliers]
        dr_c_in = [dR_cam[i] for i in best_inliers]
        best_R = solve_rotation_handeye(dr_r_in, dr_c_in)

    logger.info(f"RANSAC rotation: {len(best_inliers)}/{n} inliers")
    return best_R, best_inliers


# ============================================================
# Steps B + C with outlier frames, and the full pipeline
# ============================================================


def _nearest_pairs(times_ref, times_cam, dt):
    """Nearest camera sample to each reference time after shifting the camera times by dt.

    Returns ([(i, j, |t_cam[j] + dt - t_ref[i]|) for every i], max_td): j is the first
    nearest index; max_td is the match tolerance, 0.6 x the median sample spacing of both
    streams (0.1 s without one). Callers keep their own accept rule against max_td.
    """
    all_dts = []
    if len(times_ref) > 1: all_dts.extend(np.diff(times_ref).tolist())
    if len(times_cam) > 1: all_dts.extend(np.diff(times_cam).tolist())
    max_td = np.median(all_dts) * 0.6 if all_dts else 0.1

    corrected = np.asarray(times_cam) + dt
    pairs = []
    for i, t_ref in enumerate(times_ref):
        diffs = np.abs(corrected - t_ref)
        j = np.argmin(diffs)
        pairs.append((i, j, diffs[j]))
    return pairs, max_td


def _match_relative_rotations_with_indices(dR_ref, times_ref, dR_cam, times_cam, dt):
    """Match relative rotations and return matched indices too."""
    pairs, max_td = _nearest_pairs(times_ref, times_cam, dt)
    matched = []
    idx_ref = []
    idx_cam = []
    for i, j, d in pairs:
        if d < max_td:
            matched.append((dR_ref[i], dR_cam[j]))
            idx_ref.append(i)
            idx_cam.append(j)
    return matched, idx_ref, idx_cam


def estimate_rotation_with_outliers(poses_ref, ts_ref, poses_cam, ts_cam, dt):
    """Estimate R_cam_ref and return raw timestamps of outlier frames.

    If the initial RANSAC inlier rate is below 70%, retries with more
    iterations (1000) and a tighter threshold (0.05 rad ≈ 2.9°).  This
    catches cases where cross-model cameras or inaccurate dt produce
    many soft outliers that confuse the default threshold.
    """
    dR_ref, times_ref = compute_relative_rotations(poses_ref, ts_ref)
    dR_cam, times_cam = compute_relative_rotations(poses_cam, ts_cam)

    matched, idx_ref, idx_cam = _match_relative_rotations_with_indices(
        dR_ref, times_ref, dR_cam, times_cam, dt)

    if len(matched) < 3:
        return gtsam.Rot3(), set(), set()

    dR_r = [m[0] for m in matched]
    dR_c = [m[1] for m in matched]

    R, inliers = solve_rotation_ransac(dR_r, dR_c)
    inlier_rate = len(inliers) / len(matched)

    # Retry with more iterations and tighter threshold if inlier rate is low
    if inlier_rate < 0.7 and len(matched) >= 20:
        logger.warning(
            f"Low rotation inlier rate ({inlier_rate:.1%}), retrying with "
            f"1000 iterations and tighter threshold (0.05 rad)")
        R2, inliers2 = solve_rotation_ransac(
            dR_r, dR_c, n_iter=1000, threshold=0.05)
        inlier_rate2 = len(inliers2) / len(matched)
        if inlier_rate2 > inlier_rate:
            R, inliers = R2, inliers2
            inlier_rate = inlier_rate2
            logger.info(f"Retry improved inlier rate to {inlier_rate:.1%}")
        else:
            logger.info(f"Retry did not improve ({inlier_rate2:.1%} vs {inlier_rate:.1%})")

    if inlier_rate < 0.5:
        logger.warning(
            f"Rotation alignment poor: {inlier_rate:.1%} inliers. "
            f"Extrinsic R may be unreliable.")

    # Relative rotation pair k involves frames k and k+1.  Convert the
    # pair-local indices to raw timestamps immediately: integer positions are
    # not stable after the PnP, rate, and quality filters used by later stages.
    outlier_ref_timestamps = set()
    outlier_cam_timestamps = set()
    inlier_set = set(inliers)
    for k in range(len(idx_ref)):
        if k not in inlier_set:
            outlier_ref_timestamps.add(float(ts_ref[idx_ref[k]]))
            outlier_ref_timestamps.add(float(ts_ref[idx_ref[k] + 1]))
            outlier_cam_timestamps.add(float(ts_cam[idx_cam[k]]))
            outlier_cam_timestamps.add(float(ts_cam[idx_cam[k] + 1]))

    angle = np.linalg.norm(gtsam.Rot3.Logmap(R)) * 180 / np.pi
    logger.info(f"Rotation alignment: {len(inliers)}/{len(matched)} inliers "
                f"({inlier_rate:.1%}), angle={angle:.2f}deg, "
                f"outlier frames: ref={len(outlier_ref_timestamps)}, "
                f"cam={len(outlier_cam_timestamps)}")
    return R, outlier_ref_timestamps, outlier_cam_timestamps


# Scale factor from the median absolute deviation to a Gaussian sigma.
_MAD_TO_SIGMA = 1.4826


def _adaptive_translation_threshold(t_arr, floor, k_sigma):
    """Inlier radius for the translation RANSAC, scaled from the data.

    Per-axis robust sigmas (1.4826 * MAD about the coordinate median) are
    combined into a total spread ``sqrt(sum sigma_a^2)``; the radius is
    ``k_sigma`` times that spread, never below ``floor``.  PnP depth noise and
    inter-camera match gaps make the per-frame samples scatter by several cm
    on low-resolution fisheye rigs or distant boards, so a fixed absolute
    radius would reject most genuine inliers there.  The MAD needs a majority
    of consistent samples, which holds for a rigid extrinsic once PnP
    outliers have been removed.
    """
    center = np.median(t_arr, axis=0)
    sigma_axes = _MAD_TO_SIGMA * np.median(np.abs(t_arr - center), axis=0)
    spread = float(np.sqrt(np.sum(sigma_axes ** 2)))
    return max(float(floor), float(k_sigma) * spread)


def estimate_translation_with_outliers(poses_ref, ts_ref, poses_cam, ts_cam,
                                         R_cam_ref, dt, n_iter=200, threshold=0.02,
                                         adaptive=True, k_sigma=2.5,
                                         min_inlier_fraction=0.5):
    """Estimate t_cam_ref and return raw timestamps of outlier frames.

    ``threshold`` is the inlier radius in metres.  With ``adaptive=True``
    (default) it is a floor and the radius grows with the robust spread of
    the samples (see ``_adaptive_translation_threshold``).  If the best
    consensus covers less than ``min_inlier_fraction`` of the matched
    samples, the rejection is not trusted: a warning is logged and the
    median of all samples is returned with no outlier frames.
    """
    pairs, max_td = _nearest_pairs(ts_ref, ts_cam, dt)

    R_mat = R_cam_ref.matrix()
    t_estimates = []
    match_ref_timestamps = []
    match_cam_timestamps = []

    for i, j, d in pairs:
        if d > max_td:
            continue
        T_ref_board = poses_ref[i].inverse()
        T_cam_board = poses_cam[j].inverse()
        t_r = np.array(T_ref_board.translation())
        t_c = np.array(T_cam_board.translation())
        t_estimates.append(t_c - R_mat @ t_r)
        # Keep each camera's raw timestamp, not the corrected timestamp and
        # not the position in this filtered trajectory.
        match_ref_timestamps.append(float(ts_ref[i]))
        match_cam_timestamps.append(float(ts_cam[j]))

    if not t_estimates:
        return np.zeros(3), set(), set()

    t_arr = np.array(t_estimates)
    if len(t_arr) < 3:
        return np.median(t_arr, axis=0), set(), set()

    # RANSAC
    if n_iter <= 0:
        raise ValueError("n_iter must be positive")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive")

    if adaptive:
        threshold = _adaptive_translation_threshold(t_arr, threshold, k_sigma)

    best_t = np.median(t_arr, axis=0)
    best_mask = np.zeros(len(t_arr), dtype=bool)
    best_count = 0
    best_score = np.inf

    def _refine_consensus(seed_mask):
        """Iterate median/reclassification until the inlier mask is stable."""
        mask = seed_mask.copy()
        seen_masks = set()
        for _ in range(len(t_arr) + 2):
            if not np.any(mask):
                return None, mask
            mask_key = mask.tobytes()
            if mask_key in seen_masks:
                # A cycle is unlikely for a coordinate median, but return a
                # model and the mask classified by that exact model rather
                # than allowing their identities to disagree.
                refined_t = np.median(t_arr[mask], axis=0)
                classified = np.linalg.norm(
                    t_arr - refined_t, axis=1) < threshold
                return refined_t, classified
            seen_masks.add(mask_key)

            refined_t = np.median(t_arr[mask], axis=0)
            classified = np.linalg.norm(
                t_arr - refined_t, axis=1) < threshold
            if np.array_equal(classified, mask):
                return refined_t, mask
            mask = classified

        # Bound the work even for a pathological sequence of masks.  Keep the
        # returned identity consistent with this exact model.
        if not np.any(mask):
            return None, mask
        refined_t = np.median(t_arr[mask], axis=0)
        classified = np.linalg.norm(t_arr - refined_t, axis=1) < threshold
        return refined_t, classified

    rng = random.Random(0)  # seed RANSAC for run-to-run reproducibility
    for _ in range(n_iter):
        idx = rng.randint(0, len(t_arr) - 1)
        t_cand = t_arr[idx]
        dists = np.linalg.norm(t_arr - t_cand, axis=1)
        mask = dists < threshold
        count = int(np.sum(mask))
        if count == 0:
            continue

        # Refine from the candidate consensus, reclassifying until the model
        # and mask describe the same sample set.
        refined_t, refined_mask = _refine_consensus(mask)
        if refined_t is None or not np.any(refined_mask):
            continue
        refined_dists = np.linalg.norm(t_arr - refined_t, axis=1)
        refined_count = int(np.sum(refined_mask))
        refined_score = float(np.median(refined_dists[refined_mask]))
        if (refined_count > best_count
                or (refined_count == best_count
                    and refined_score < best_score)):
            best_count = refined_count
            best_score = refined_score
            best_mask = refined_mask
            best_t = refined_t

    if best_count < min_inlier_fraction * len(t_arr):
        # A tiny consensus means the radius does not match the sample
        # scatter, not that most frames are wrong; flagging them all would
        # downweight most of the observations in Stage 4.
        t_all = np.median(t_arr, axis=0)
        logger.warning(
            f"RANSAC translation: consensus {best_count}/{len(t_arr)} below "
            f"{min_inlier_fraction:.0%} at threshold {threshold * 100:.1f} cm; "
            f"keeping all samples (t=[{t_all[0]:.4f}, {t_all[1]:.4f}, "
            f"{t_all[2]:.4f}])")
        return t_all, set(), set()

    # Collect outlier frames
    outlier_ref_timestamps = set()
    outlier_cam_timestamps = set()
    for k in range(len(t_arr)):
        if not best_mask[k]:
            outlier_ref_timestamps.add(match_ref_timestamps[k])
            outlier_cam_timestamps.add(match_cam_timestamps[k])

    n_inliers = int(np.sum(best_mask))
    logger.info(f"RANSAC translation: {n_inliers}/{len(t_arr)} inliers "
                f"(threshold {threshold * 100:.1f} cm), "
                f"t=[{best_t[0]:.4f}, {best_t[1]:.4f}, {best_t[2]:.4f}], "
                f"outlier frames: ref={len(outlier_ref_timestamps)}, "
                f"cam={len(outlier_cam_timestamps)}")
    return best_t, outlier_ref_timestamps, outlier_cam_timestamps


def align_cameras(trajectories, timestamps, reference_camera=0,
                    all_frames=None, image_sizes=None,
                    skip_temporal=False):
    """Align all camera trajectories to the reference camera.

    Pipeline:
      0. PnP outlier detection and removal
      A. Temporal alignment:
         - Try the homography method (no intrinsics) if ``all_frames``
           and ``image_sizes`` are provided.
         - Fall back to PnP-based angular velocity cross-correlation.
      B. Rotation alignment (RANSAC hand-eye)
      C. Translation alignment (RANSAC)

    Args:
        trajectories: cam_id → list[Pose3] (from PnP).
        timestamps: cam_id → list[float].
        reference_camera: int.
        all_frames: Optional cam_id → list[FrameObservations] for the homography method
                    homography-based time offset (no intrinsics needed).
        image_sizes: Optional cam_id → (width, height) for the homography method.

    Returns:
        cam_id -> (dt, T_cam_i_ref, outlier_frame_timestamps)
    """
    results = {}

    poses_ref = trajectories[reference_camera]
    ts_ref = timestamps[reference_camera]

    inlier_ref = detect_pnp_outliers(poses_ref, ts_ref)
    poses_ref_clean, ts_ref_clean = filter_trajectory(poses_ref, ts_ref, inlier_ref)

    # Convert PnP-local indices to raw timestamps before filtering changes the
    # list identity again.
    ref_pnp_outliers = {
        float(ts_ref[i]) for i, v in enumerate(inlier_ref) if not v
    }

    for cam_id in trajectories:
        if cam_id == reference_camera:
            continue

        poses_cam = trajectories[cam_id]
        ts_cam = timestamps[cam_id]

        inlier_cam = detect_pnp_outliers(poses_cam, ts_cam)
        poses_cam_clean, ts_cam_clean = filter_trajectory(poses_cam, ts_cam, inlier_cam)
        cam_pnp_outliers = {
            float(ts_cam[i]) for i, v in enumerate(inlier_cam) if not v
        }

        logger.info(f"Aligning camera {cam_id}: "
                     f"ref={len(poses_ref_clean)}/{len(poses_ref)} inliers, "
                     f"cam={len(poses_cam_clean)}/{len(poses_cam)} inliers")

        # Step A — time offset estimation.
        # Skip entirely when temporal alignment is not needed (no_td=true,
        # synchronized cameras, or image-folder datasets with fake timestamps).
        if skip_temporal:
            dt_pnp = 0.0
            dt_m5 = 0.0
            logger.info(f"  Skipping temporal alignment (no time offset)")
        else:
            # Run BOTH methods and pick the one with stronger correlation:
            #   1. PnP-based: uses Stage 2 optimized trajectory (has intrinsics)
            #   2. Homography: uses raw 2D tags only (no intrinsics)
            # The PnP method uses Stage 2's GTSAM-optimized poses which are
            # accurate for same-model stereo. Homography is more robust for cross-FOV
            # cameras where PnP quality differs between cameras.

            # Method 1: PnP angular velocity (Stage 2 optimized trajectory)
            dt_pnp = estimate_time_offset_rotation(
                poses_ref_clean, ts_ref_clean,
                poses_cam_clean, ts_cam_clean)

            # Method 2: homography (no intrinsics, if frames available)
            dt_m5 = 0.0
            if (all_frames is not None and image_sizes is not None
                    and reference_camera in all_frames and cam_id in all_frames
                    and reference_camera in image_sizes and cam_id in image_sizes):
                dt_m5 = estimate_time_offset_homography(
                    all_frames[reference_camera], image_sizes[reference_camera],
                    all_frames[cam_id], image_sizes[cam_id])

        # Pick: if both agree (within 50ms), use PnP (more precise with
        # good intrinsics). If they disagree, use homography (more robust).
        if abs(dt_pnp) < 0.001 and abs(dt_m5) < 0.001:
            dt = 0.0  # both say synced
        elif abs(dt_pnp) < 0.001:
            dt = dt_m5  # PnP failed, homography found something
            logger.info(f"  dt: PnP=0, homography={dt_m5*1000:.1f}ms → using homography")
        elif abs(dt_m5) < 0.001:
            dt = dt_pnp  # homography failed, PnP found something
            logger.info(f"  dt: PnP={dt_pnp*1000:.1f}ms, homography=0 → using PnP")
        elif abs(dt_pnp - dt_m5) < 0.05:
            # Both agree within 50ms — use PnP (more precise from Stage 2 trajectory)
            dt = dt_pnp
            logger.info(f"  dt: PnP={dt_pnp*1000:.1f}ms, homography={dt_m5*1000:.1f}ms → "
                         f"agree, using PnP (Stage 2 trajectory)")
        else:
            # Disagree — use homography (more robust, no intrinsic dependency)
            dt = dt_m5
            logger.info(f"  dt: PnP={dt_pnp*1000:.1f}ms, homography={dt_m5*1000:.1f}ms → "
                         f"disagree, using homography (intrinsic-free)")
        # Step B — with outlier tracking
        R, rot_outlier_ref, rot_outlier_cam = estimate_rotation_with_outliers(
            poses_ref_clean, ts_ref_clean, poses_cam_clean, ts_cam_clean, dt)

        # Step C — with outlier tracking
        t, trans_outlier_ref, trans_outlier_cam = estimate_translation_with_outliers(
            poses_ref_clean, ts_ref_clean, poses_cam_clean, ts_cam_clean, R, dt)

        T = gtsam.Pose3(R, gtsam.Point3(t[0], t[1], t[2]))

        # Combine all outlier sources for this camera
        cam_outliers = cam_pnp_outliers | rot_outlier_cam | trans_outlier_cam
        # Also track ref outliers (shared across all non-ref cameras)
        ref_outliers = ref_pnp_outliers | rot_outlier_ref | trans_outlier_ref

        # Store combined outlier set per camera
        outlier_frame_timestamps = {
            reference_camera: ref_outliers,
            cam_id: cam_outliers,
        }

        results[cam_id] = (dt, T, outlier_frame_timestamps)

        baseline = np.linalg.norm(t)
        angle = np.linalg.norm(gtsam.Rot3.Logmap(R)) * 180 / np.pi
        logger.info(f"Camera {cam_id}: dt={dt:.6f}s, R={angle:.2f}deg, baseline={baseline:.4f}m, "
                     f"outlier frames: ref={len(ref_outliers)}, cam={len(cam_outliers)}")

    return results
