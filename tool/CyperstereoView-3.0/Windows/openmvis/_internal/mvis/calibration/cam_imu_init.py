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
"""Camera-IMU spatial-temporal initialization.

ARCHITECTURE: camera calibration and camera-IMU calibration are TWO
SEPARATE PROCESSES. The camera calibration (its own recording) delivers only
the camera **intrinsics + distortion**; the cam-IMU calibration is a separate
process whose inputs are those intrinsics, the board detections of the cam-IMU
recording, and the raw IMU stream. The init therefore builds its own pose
track: per-frame PnP (`pnp_pose_track`, stage I0) using the provided
intrinsics — there are no BA-refined poses to inherit (board poses of the
cam-IMU sequence are first jointly refined by the joint solve itself).

Stages:

  I0  pose track            — per-frame PnP from detections + given intrinsics
  I1  time offset td        — cross-correlation of |ω| (rotation- & scale-free)
  I2  R_CtoI + gyro bias    — centered Kabsch on angular velocities
  I3  s, gravity, v_k, p, b_a — SCALE-UNKNOWN linear visual-inertial alignment
                              (full VINS-Mono system; board mode's s ≈ 1 and
                              ‖g_LS‖ ≈ 9.81 are consistency gates)
  I4  gravity-aligned world + `CamImuSeed` assembly

Conventions (match the factors / simulator):
  * td: a camera frame stamped `t` is captured at trajectory/IMU time `t + td`
    (same convention as `CameraCalibImuFactor` / `simulate_camera`).
  * IMU intrinsics are treated as IDENTITY at init (refined in the joint solve).
  * world `W` is gravity-aligned, `g_inW = [0, 0, -9.81]` (z-up).
  * frame chain: `T_ItoB = T_CtoB · T_CtoI⁻¹`, so `R_ItoB = R_CtoB · R_CtoIᵀ`
    and `p_IinB = s·p̄_CinB + R_CtoB · p_IinC`.

Designed for LONG windows (40-60 s — offline calibration takes the excitation);
the alignment runs on decimated keyframes so the LS stays small.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mvis.core.gravity import STANDARD_MVIS
import numpy as np
import gtsam

from mvis.core.camera_models import canonical_dist, legacy_model
from mvis.calibration.factors.lie_groups import exp_so3, log_so3
from mvis.imu import ImuModel, RPNG
from mvis.calibration.factors.aci import ACIPreintegrator

GRAVITY_MAG = STANDARD_MVIS   # the default of the init functions; a solve passes its cfg.gravity


# ---------------------------------------------------------------------------
# I0 — per-frame PnP pose track (detections + given intrinsics ONLY)
# ---------------------------------------------------------------------------
def pnp_pose_track(t_cam, ids_list, uv_list, board_points, cal_vec,
                   is_fisheye: bool = False, min_corners: int = 6,
                   model: str | None = None, pnp_method: str = "iterative"):
    """Per-frame PnP: board detections + camera intrinsics → `T_CtoB` track.

    The cam-IMU process receives only the camera intrinsics + distortion from
    the (separate) camera calibration — the pose track is built here, not
    inherited. Frames with < `min_corners` corners or failed PnP are dropped.

    Args:
        t_cam: (F,) frame stamps.
        ids_list / uv_list: per frame, corner IDs and (n,2) pixels.
        board_points: (N,3) board-frame corner positions (z = 0 plane).
        cal_vec: intrinsics [fx, fy, cx, cy, dist...] (length = model n_intr).
        is_fisheye: back-compat flag (equidistant); ignored when `model` given.
        model: distortion model. Radtan uses OpenCV solvePnP directly; every other
            model goes through `initializer.board_pose_pnp`, the camera-only init's
            solve: bearings from its camera model (pixels with no ray or past 80 deg
            dropped), RANSAC + planar IPPE candidates, the best by true-model pixel
            error. Defaults from `is_fisheye`.
        pnp_method: "iterative" (legacy) or "ippe". ITERATIVE can diverge to
            absurd poses or flip between the two planar-ambiguity basins on
            planar boards (observed on Monado MOC cam0: km-scale translations
            that poison the alignment LS); IPPE resolves the planar two-fold
            ambiguity by construction. IPPE falls back to ITERATIVE per frame
            when it declines (needs ≥4 non-degenerate coplanar points).

    Returns (t_kept (K,), poses [gtsam.Pose3 T_CtoB], n_corners (K,)), n_corners being
    the corners the PnP used.
    """
    import cv2

    cal_vec = np.asarray(cal_vec, float)
    if model is None:
        model = legacy_model(is_fisheye, cal_vec)
    K_mat = np.array([[cal_vec[0], 0, cal_vec[2]],
                      [0, cal_vec[1], cal_vec[3]],
                      [0, 0, 1.0]])
    dist = (np.asarray(canonical_dist(model, cal_vec[4:]))
            if model == "radtan" else None)    # cv2 takes radtan only
    board_points = np.asarray(board_points, float)
    from mvis.calibration.initializer import board_pose_pnp
    pnp_flag = cv2.SOLVEPNP_IPPE if pnp_method == "ippe" else cv2.SOLVEPNP_ITERATIVE

    t_kept, poses, n_corners = [], [], []
    for t, ids, uv in zip(np.asarray(t_cam, float), ids_list, uv_list):
        if len(ids) < min_corners:
            continue
        obj = board_points[np.asarray(ids, int)].astype(np.float64)
        img = np.asarray(uv, np.float64).reshape(-1, 2)

        def _pnp(o, pts, K, d):
            if pnp_method == "ippe":
                try:
                    ok, rv, tv = cv2.solvePnP(o, pts, K, d,
                                              flags=cv2.SOLVEPNP_IPPE)
                    if ok:
                        return ok, rv, tv
                except cv2.error:
                    pass
            return cv2.solvePnP(o, pts, K, d, flags=cv2.SOLVEPNP_ITERATIVE)

        try:
            if model == "radtan":
                ok, rvec, tvec = _pnp(obj, img, K_mat, dist)
                used = len(ids)
            else:                       # bearings: the camera-only initializer's solve
                pose = board_pose_pnp(obj, img, K_mat, cal_vec[4:], model, pnp_flag,
                                      min_corners, 0.0)
                ok = pose is not None
                if ok:
                    rvec, tvec, used = pose
        except cv2.error:
            continue
        if not ok or not (np.isfinite(rvec).all() and np.isfinite(tvec).all()):
            continue
        R_BtoC, _ = cv2.Rodrigues(rvec)
        T_BtoC = np.eye(4)
        T_BtoC[:3, :3] = R_BtoC
        T_BtoC[:3, 3] = tvec.ravel()
        poses.append(gtsam.Pose3(np.linalg.inv(T_BtoC)))   # T_CtoB
        t_kept.append(float(t))
        n_corners.append(used)

    # outlier rejection: planar-PnP ambiguity flips produce isolated pose
    # excursions (rotation AND translation velocity jumps vs neighbors)
    from mvis.calibration.pose_alignment import (
        detect_pnp_outliers, filter_trajectory)
    if not poses:
        raise ValueError("pnp_pose_track: no valid PnP poses (no board frames in "
                         "window? check intrinsics / board size / time window)")
    inlier = np.asarray(detect_pnp_outliers(poses, t_kept), dtype=bool)
    inlier &= _translation_jump_inliers(t_kept, poses)
    poses, t_kept = filter_trajectory(poses, t_kept, list(inlier))
    n_corners = np.array(n_corners, int)[inlier]
    return np.array(t_kept), poses, n_corners


def _translation_jump_inliers(t, poses, multiplier: float = 3.0,
                              v_min: float = 2.0):
    """Translation analog of `detect_pnp_outliers`: flag poses whose linear
    velocity to either neighbor exceeds multiplier × 95th percentile."""
    n = len(poses)
    if n < 3:
        return np.ones(n, bool)
    v = np.array([np.linalg.norm(
        np.asarray(poses[i + 1].translation(), float)
        - np.asarray(poses[i].translation(), float)) / max(t[i + 1] - t[i], 1e-9)
        for i in range(n - 1)])
    thr = max(np.percentile(v, 95) * multiplier, v_min)
    from mvis.calibration.pose_alignment import velocity_jump_inliers
    return velocity_jump_inliers(v, thr)


# ---------------------------------------------------------------------------
# signal helpers
# ---------------------------------------------------------------------------
def camera_angular_velocity(t_cam, R_CtoB_list, stride: int = 1):
    """Interval-mean camera angular velocity, ω_C = Log(R_kᵀ·R_{k+stride})/Δt
    (camera frame). `stride` sets the differentiation baseline: longer
    baselines average down per-frame PnP noise (which differentiation
    amplifies) while the interval-mean gyro pairing stays exact at any
    baseline — use ~0.3 s for the td/rotation stages under real noise.

    Returns (t_mid (K,), w_C (K, 3), pairs (K, 2)) — pairs are the interval
    endpoints (t_k, t_{k+stride}) the gyro side must integrate over.
    """
    t_cam = np.asarray(t_cam, float)
    t_mid, w, pairs = [], [], []
    for k in range(len(t_cam) - stride):
        dt = t_cam[k + stride] - t_cam[k]
        if dt <= 0:
            continue
        dR = R_CtoB_list[k].T @ R_CtoB_list[k + stride]
        w.append(log_so3(dR) / dt)
        t_mid.append(0.5 * (t_cam[k] + t_cam[k + stride]))
        pairs.append((t_cam[k], t_cam[k + stride]))
    return np.array(t_mid), np.array(w).reshape(-1, 3), np.array(pairs)


def _stride_for_baseline(t_cam, baseline: float):
    """Frame stride whose differentiation interval is ≈ `baseline` seconds."""
    dt = float(np.median(np.diff(np.asarray(t_cam, float))))
    return max(1, int(round(baseline / max(dt, 1e-6))))


def _interp_vec(t_query, t, v):
    """Per-axis linear interpolation of a (N,3) signal."""
    return np.column_stack([np.interp(t_query, t, v[:, i]) for i in range(3)])


def _cumulative_rotation(t_imu, w_m):
    """Gyro dead-reckoned rotations R_cum[i] from sample 0 to sample i
    (trapezoidal ω). Bias is NOT removed — the Kabsch centering handles it."""
    R = np.empty((len(t_imu), 3, 3))
    R[0] = np.eye(3)
    for i in range(len(t_imu) - 1):
        dt = t_imu[i + 1] - t_imu[i]
        R[i + 1] = R[i] @ exp_so3(0.5 * (w_m[i] + w_m[i + 1]) * dt)
    return R


def _gyro_interval_omega(t_imu, R_cum, w_m, t0, t1):
    """Mean gyro angular velocity over [t0, t1] via the integrated rotation —
    the consistent counterpart of the camera's Log(R_kᵀR_{k+1})/Δt (pairing an
    interval average with instantaneous gyro samples biases td by ~ms)."""
    def rot_at(tq):
        i = int(np.clip(np.searchsorted(t_imu, tq) - 1, 0, len(t_imu) - 2))
        return R_cum[i] @ exp_so3(w_m[i] * (tq - t_imu[i]))
    return log_so3(rot_at(t0).T @ rot_at(t1)) / (t1 - t0)


def _gyro_interval_signal(pairs, t_imu, R_cum, w_m, td):
    """Interval-mean gyro ω over each camera interval (a, b) shifted by td."""
    return np.array([
        _gyro_interval_omega(t_imu, R_cum, w_m, a + td, b + td)
        for a, b in pairs])


# ---------------------------------------------------------------------------
# I1 — time offset
# ---------------------------------------------------------------------------
def estimate_time_offset_coarse(t_cam, R_CtoB_list, t_imu, w_m,
                                max_offset: float = 0.1, grid_hz: float = 500.0):
    """COARSE td by cross-correlating |ω_cam| against |ω_gyro| (mean-centered).

    Convention: a frame stamped `t` is captured at IMU time `t + td`, so the
    camera signal at stamp t matches the gyro signal at t + td.

    The |ω| autocorrelation of smooth calibration motion is broad — this
    locates td only to a few ms. `refine_time_offset` (LS valley with R/b_g
    re-fit per td) provides the precision; this provides the global search
    over ±max_offset.

    Returns (td, report) — report carries the peak quality (peak/2nd-peak
    ratio; flat or multi-peak correlation ⇒ unreliable td).
    """
    t_mid, w_C, _pairs = camera_angular_velocity(t_cam, R_CtoB_list)
    t_imu = np.asarray(t_imu, float)
    mag_C = np.linalg.norm(w_C, axis=1)
    mag_I = np.linalg.norm(np.asarray(w_m, float), axis=1)

    # common uniform grid on the overlap (camera-time axis)
    dt = 1.0 / grid_hz
    lo = max(t_mid[0], t_imu[0] + max_offset)
    hi = min(t_mid[-1], t_imu[-1] - max_offset)
    if hi - lo < 1.0:
        raise ValueError("insufficient overlap between camera and IMU signals")
    grid = np.arange(lo, hi, dt)
    # mask out grid points inside board-visibility gaps — interpolating the
    # camera signal across a blind gap fabricates plateaus that bias the peak
    pos = np.searchsorted(t_mid, grid)
    d_left = np.abs(grid - t_mid[np.clip(pos - 1, 0, len(t_mid) - 1)])
    d_right = np.abs(t_mid[np.clip(pos, 0, len(t_mid) - 1)] - grid)
    gap_tol = 2.0 * float(np.median(np.diff(t_mid)))
    mask = np.minimum(d_left, d_right) <= gap_tol
    # high-pass both signals (moving-average subtraction, ~1 s window): kills
    # slow trends that tilt the broad correlation and drag the peak to the
    # search boundary under noise
    k_hp = int(round(1.0 / dt)) | 1
    kern = np.ones(k_hp) / k_hp

    def _hp(x):
        return x - np.convolve(x, kern, mode="same")

    full_grid = grid
    a_full = _hp(np.interp(full_grid, t_mid, mag_C))
    grid = full_grid[mask]
    a = a_full[mask]
    a -= a.mean()

    n_lag = int(round(max_offset / dt))
    lags = np.arange(-n_lag, n_lag + 1)
    corr = np.empty(len(lags))
    for i, lag in enumerate(lags):
        b = _hp(np.interp(full_grid + lag * dt, t_imu, mag_I))[mask]
        b -= b.mean()
        corr[i] = a @ b
    i_pk = int(np.argmax(corr))

    # sub-sample parabolic refinement
    if 0 < i_pk < len(lags) - 1:
        y0, y1, y2 = corr[i_pk - 1], corr[i_pk], corr[i_pk + 1]
        denom = (y0 - 2 * y1 + y2)
        frac = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    else:
        frac = 0.0
    td = (lags[i_pk] + frac) * dt

    # peak quality: best vs best-outside-neighborhood
    excl = max(1, int(round(0.02 / dt)))           # 20 ms exclusion zone
    mask = np.abs(lags - lags[i_pk]) > excl
    second = np.max(corr[mask]) if np.any(mask) else 0.0
    peak_ratio = corr[i_pk] / second if second > 0 else np.inf
    report = {"td": td, "peak_ratio": float(peak_ratio),
              "corr_peak": float(corr[i_pk])}
    return td, report


# ---------------------------------------------------------------------------
# I2 — extrinsic rotation + gyro bias (also the td-refinement objective)
# ---------------------------------------------------------------------------
def _kabsch(w_C, w_I):
    """Centered Kabsch: w_I ≈ R·w_C + b_g. Returns (R, b_g, rms, svals)."""
    mC, mI = w_C.mean(0), w_I.mean(0)
    wC_c, wI_c = w_C - mC, w_I - mI
    H = wC_c.T @ wI_c                                  # maximize tr(H·R)
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(V @ U.T))])
    R = V @ D @ U.T
    b_g = mI - R @ mC
    resid = w_I - b_g - w_C @ R.T
    return R, b_g, float(np.sqrt(np.mean(resid ** 2))), S / max(len(w_C), 1)


def estimate_rotation_gyro(t_cam, R_CtoB_list, t_imu, w_m, td: float,
                           stride: int = 1, _precomp=None):
    """R_CtoI and b_g by centered Kabsch on INTERVAL-MEAN angular velocities.

    Camera ω is an interval average Log(R_kᵀR_{k+1})/Δt, so the gyro side uses
    the matching interval mean (Log of the integrated gyro rotation over the
    same td-shifted interval) — pairing it with instantaneous samples leaves a
    large residual floor and biases the td refinement by ~ms. Subtracting each
    signal's mean removes the constant b_g exactly → SVD/Kabsch gives R_CtoI;
    then b_g = mean(ω_I) − R_CtoI·mean(ω_C).

    Returns (R_CtoI (3,3), b_g (3,), report) — report carries the angular-rate
    residual RMS (the td-refinement objective) and the excitation singular
    values (degeneracy guard: 1-axis rotation makes R partly unobservable).
    """
    t_imu = np.asarray(t_imu, float)
    w_m = np.asarray(w_m, float)
    if _precomp is None:
        _t_mid, w_C, pairs = camera_angular_velocity(t_cam, R_CtoB_list, stride)
        R_cum = _cumulative_rotation(t_imu, w_m)
        _precomp = (pairs, w_C, R_cum)
    pairs, w_C, R_cum = _precomp
    w_I = _gyro_interval_signal(pairs, t_imu, R_cum, w_m, td)

    R_CtoI, b_g, rms, svals = _kabsch(w_C, w_I)
    report = {"residual_rms": rms, "excitation_svals": svals}
    return R_CtoI, b_g, report


def refine_time_offset(t_cam, R_CtoB_list, t_imu, w_m, td0: float,
                       half_window: float = 0.1, step: float = 4e-3,
                       stride: int = 1, return_precomp: bool = False):
    """td by minimizing the Kabsch residual RMS over a td grid (R and b_g
    re-fit per candidate): a wide pass over the FULL search range, then fine
    passes + parabolic sub-step around its minimum. The 3-D LS valley is the
    robust objective — the broad |ω| correlation peak can be tens of ms off
    (or hit the search boundary) on smooth/noisy data, so `td0` only centers
    the window; `half_window` defaults to the full ±search range. Under PnP
    noise use a longer differentiation baseline (`stride`).

    `return_precomp=True` also returns the `(pairs, w_C, R_cum)` tuple the
    scan was built on, which `estimate_rotation_gyro(..., _precomp=...)` at
    the same `(t_cam, R_CtoB_list, t_imu, w_m, stride)` takes as is
    (`initialize_cam_imu` then skips a second `camera_angular_velocity`
    and the per-sample `_cumulative_rotation`; both are deterministic, so the
    result is bitwise the same)."""
    t_imu = np.asarray(t_imu, float)
    w_m = np.asarray(w_m, float)
    _t_mid, w_C, pairs = camera_angular_velocity(t_cam, R_CtoB_list, stride)
    R_cum = _cumulative_rotation(t_imu, w_m)
    pre = (pairs, w_C, R_cum)

    def _scan(center, half, st):
        grid = center + np.arange(-half, half + st / 2, st)
        rms = np.array([estimate_rotation_gyro(None, None, t_imu, w_m, td,
                                               _precomp=pre)[2]["residual_rms"]
                        for td in grid])
        i = int(np.argmin(rms))
        if 0 < i < len(grid) - 1:
            y0, y1, y2 = rms[i - 1], rms[i], rms[i + 1]
            denom = y0 - 2 * y1 + y2
            frac = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-15 else 0.0
        else:
            frac = 0.0
        return float(grid[i] + frac * st)

    # wide pass over the full ±half_window search range (centered on 0, the
    # td-convention origin — a boundary-stuck coarse estimate must not shift
    # the window off the truth); td0 is only used to extend the range if the
    # caller's coarse estimate falls outside it
    center = 0.0 if abs(td0) >= half_window else td0 / 2.0
    td1 = _scan(center, half_window + abs(center), step)
    td1 = _scan(td1, 1.5 * step, step / 4)              # narrow
    td2 = _scan(td1, 0.5 * step, step / 16)             # fine + parabola

    def _rms(td):
        return estimate_rotation_gyro(None, None, t_imu, w_m, td,
                                      _precomp=pre)[2]["residual_rms"]
    rms_min = _rms(td2)
    # valley sharpness: residual growth at ±half_window — the td quality gate
    # (the coarse |ω| correlation peak is inherently broad on smooth motion)
    valley_ratio = (0.5 * (_rms(td2 - half_window) + _rms(td2 + half_window))
                    / max(rms_min, 1e-12))
    report = {"rms_min": rms_min, "valley_ratio": float(valley_ratio)}
    if return_precomp:
        return td2, report, pre
    return td2, report


# ---------------------------------------------------------------------------
# I3 — scale-unknown linear alignment
# ---------------------------------------------------------------------------
def _is_monotonic(t) -> bool:
    """True when the stamps are non-decreasing (no NaN) — the precondition for
    locating a window by `searchsorted` instead of a boolean mask."""
    return bool(np.all(t[1:] >= t[:-1]))


def _slice_imu(t_imu, w_m, a_m, t0, t1):
    """IMU samples covering [t0, t1], boundary samples linearly interpolated.

    The strictly-inside samples (`t0 < t < t1`) are the half-open index range
    `[searchsorted(t, t0, 'right'), searchsorted(t, t1, 'left'))` when the
    stamps are monotonic (a speed-up: same samples in the same order as the
    mask); bag stamps can be out of order, so a non-monotonic stream keeps the
    boolean mask.
    """
    t_imu = np.asarray(t_imu, float)
    if _is_monotonic(t_imu):
        lo = int(np.searchsorted(t_imu, t0, side='right'))
        hi = max(int(np.searchsorted(t_imu, t1, side='left')), lo)
        inside = slice(lo, hi)
    else:
        inside = (t_imu > t0) & (t_imu < t1)
    t = np.concatenate([[t0], t_imu[inside], [t1]])
    w = np.vstack([_interp_vec([t0], t_imu, w_m),
                   np.asarray(w_m)[inside],
                   _interp_vec([t1], t_imu, w_m)])
    a = np.vstack([_interp_vec([t0], t_imu, a_m),
                   np.asarray(a_m)[inside],
                   _interp_vec([t1], t_imu, a_m)])
    return t, w, a


@dataclass
class AlignmentResult:
    s: float
    g_inB: np.ndarray            # refined (‖g‖ = g_mag)
    v_inB: np.ndarray            # (K+1, 3) keyframe velocities, board frame
    p_IinC: np.ndarray
    b_a: np.ndarray
    g_norm_ls: float             # ‖g‖ of the UNCONSTRAINED solve — quality gate
    residual_rms: float
    svals: np.ndarray            # singular values of the (column-scaled) LS
    estimated_ba: bool
    t_key: np.ndarray = None     # keyframes actually used (after gap filtering)
    n_skipped_intervals: int = 0  # intervals dropped by the max_interval gate
    sigma_p: np.ndarray = None   # LS 1σ of p_IinC — gate weak estimates to 0
    sigma_s: float = 0.0         # LS 1σ of the scale
    sigma_ba: np.ndarray = None  # LS 1σ of b_a (when estimated)


def linear_alignment(t_key, T_CtoB_list, R_CtoI, b_g, t_imu, w_m, a_m,
                     estimate_scale: bool = True, estimate_ba: bool = False,
                     g_mag: float = GRAVITY_MAG, sigma_g=0.01, sigma_bg=0.001,
                     sigma_a=0.02, sigma_ba=0.002, ba_sval_gate: float = 1e-3,
                     ba_sigma_gate: float = 0.05, refine_iters: int = 3,
                     max_interval: float = 1.0) -> AlignmentResult:
    """Scale-unknown linear visual-inertial alignment over keyframes (default).

    Solves x = [v_0..v_K (in B) | g_inB | p_IinC | s | b_a] from the ACI
    preintegration relations (identity IMU intrinsics, b_g from I2):

      s·p̄(k+1) + R_C(k+1)·p = s·p̄(k) + R_C(k)·p + v_k·Δt + ½·g·Δt²
                              + R_IktoB·(Δp_k + J_p_ba·b_a)
      v_(k+1) = v_k + g·Δt + R_IktoB·(Δv_k + J_v_ba·b_a)

    with R_IktoB = R_CtoB(k)·R_CtoIᵀ. The board case is the same system where
    the recovered s must be ≈ 1. ‖g‖ of the unconstrained solve is the primary
    alignment-quality gate; the returned g is refined with ‖g‖ = g_mag enforced
    (2-sphere parameterization).

    **b_a is initialized as 0 by DEFAULT** — the joint solve owns
    the accel bias; under real PnP noise its init estimate is unreliable (it
    drags gravity with it). `estimate_ba=True` keeps the columns available for
    experiments, still behind the σ gate (kept only if the conditioning
    supports them, else b_a = 0).

    Board-visibility gaps: intervals longer than `max_interval` contribute NO
    rows (IMU-only bridging across a blind gap adds noise/bias without camera
    anchoring), and keyframes with a long gap on BOTH sides are dropped.
    """
    t_key = np.asarray(t_key, float)
    T_list = list(T_CtoB_list)
    assert len(T_list) == len(t_key)
    dts = np.diff(t_key)
    ok = dts <= max_interval
    keep = np.concatenate([[False], ok]) | np.concatenate([ok, [False]])
    t_key = t_key[keep]
    T_list = [T for T, kp in zip(T_list, keep) if kp]
    K = len(t_key) - 1
    assert K >= 2, "too few keyframes after gap filtering"
    valid = np.flatnonzero(np.diff(t_key) <= max_interval)   # intervals w/ rows
    M = len(valid)

    R_C = [np.asarray(T.rotation().matrix(), float) for T in T_list]
    p_C = [np.asarray(T.translation(), float) for T in T_list]
    R_IB = [Rc @ R_CtoI.T for Rc in R_C]

    # preintegrate each VALID interval (identity intrinsics, b_g from I2, b_a=0)
    intr = ImuModel(model=RPNG)               # defaults = identity
    pres = {}
    for k in valid:
        t, w, a = _slice_imu(t_imu, w_m, a_m, t_key[k], t_key[k + 1])
        # only D_*, J_*_ba and P_meas are read below: no intrinsic / t_ga
        # Jacobian accumulation (bitwise on what is read)
        pre = ACIPreintegrator(sigma_g, sigma_bg, sigma_a, sigma_ba,
                               model=RPNG, avg_meas=True, track_intrinsics=False)
        pre.reset(np.asarray(b_g, float), np.zeros(3))
        pre.set_intrinsics(intr)
        pre.feed_imu_interval(t, w, a)
        pres[k] = pre

    nv = 3 * (K + 1)
    iv = lambda k: slice(3 * k, 3 * k + 3)
    ig = slice(nv, nv + 3)
    ip = slice(nv + 3, nv + 6)
    i_s = nv + 6
    iba = slice(nv + 7, nv + 10)
    ncols = nv + 7 + (3 if estimate_ba else 0)
    I3 = np.eye(3)

    def _build_raw():
        """UNWEIGHTED rows (3 position + 3 velocity per VALID interval) +
        per-row-type preintegration variances."""
        A = np.zeros((6 * M, ncols))
        b = np.zeros(6 * M)
        var_pos = np.zeros(M)
        var_vel = np.zeros(M)
        for j, k in enumerate(valid):
            pre = pres[k]
            Dt = pre.Dt
            var_pos[j] = max(np.trace(pre.P_meas[3:6, 3:6]) / 3, 1e-12)
            var_vel[j] = max(np.trace(pre.P_meas[6:9, 6:9]) / 3, 1e-12)
            r0, r1 = slice(6 * j, 6 * j + 3), slice(6 * j + 3, 6 * j + 6)
            # position rows
            A[r0, iv(k)] = -Dt * I3
            A[r0, ig] = -0.5 * Dt ** 2 * I3
            A[r0, ip] = R_C[k + 1] - R_C[k]
            if estimate_scale:
                A[r0, i_s] = p_C[k + 1] - p_C[k]
                b[r0] = R_IB[k] @ pre.D_p
            else:
                b[r0] = R_IB[k] @ pre.D_p - (p_C[k + 1] - p_C[k])
            if estimate_ba:
                A[r0, iba] = -(R_IB[k] @ pre.J_p_ba)
            # velocity rows
            A[r1, iv(k + 1)] = I3
            A[r1, iv(k)] = -I3
            A[r1, ig] = -Dt * I3
            if estimate_ba:
                A[r1, iba] = -(R_IB[k] @ pre.J_v_ba)
            b[r1] = R_IB[k] @ pre.D_v
        return A, b, var_pos, var_vel

    A0, b0, var_pos, var_vel = _build_raw()
    pos_rows = np.zeros(6 * M, bool)
    for j in range(M):
        pos_rows[6 * j:6 * j + 3] = True

    def _weights(cam_pos_var):
        """Row weights: position rows carry preint + CAMERA position noise —
        the dominant error under real PnP (weighting them by the tiny preint
        covariance alone makes the LS chase noisy camera positions and
        collapses the scale)."""
        w = np.empty(6 * M)
        for j in range(M):
            w[6 * j:6 * j + 3] = 1.0 / np.sqrt(var_pos[j] + 2.0 * cam_pos_var)
            w[6 * j + 3:6 * j + 6] = 1.0 / np.sqrt(var_vel[j])
        return w

    def _solve(cam_pos_var, drop_ba=False):
        w = _weights(cam_pos_var)
        A, b = A0 * w[:, None], b0 * w
        # conditioning gate on the column-scaled system (units differ per column)
        scale = np.maximum(np.linalg.norm(A, axis=0), 1e-12)
        svals = np.linalg.svd(A / scale, compute_uv=False)
        estimated_ba = estimate_ba and not drop_ba
        ncols_used = ncols
        if estimate_ba and not drop_ba and svals[-1] < ba_sval_gate:
            estimated_ba = False
        if not estimated_ba:
            # drop the b_a columns — honest zero beats a poorly-conditioned guess
            A = A[:, :nv + 7]
            ncols_used = nv + 7
        x, *_ = np.linalg.lstsq(A[:, :ncols_used], b, rcond=None)

        g_ls = x[ig]
        g_norm_ls = float(np.linalg.norm(g_ls))

        # gravity refinement: enforce ‖g‖ = g_mag on the 2-sphere tangent
        g_hat = g_ls / max(g_norm_ls, 1e-9)
        for _ in range(refine_iters):
            b1 = np.cross(g_hat, [1.0, 0.0, 0.0])
            if np.linalg.norm(b1) < 0.1:
                b1 = np.cross(g_hat, [0.0, 1.0, 0.0])
            b1 /= np.linalg.norm(b1)
            b2 = np.cross(g_hat, b1)
            B = np.column_stack([b1, b2])              # 3×2 tangent basis
            # substitute g = g_mag·ĝ + B·w into the LS
            A2 = np.delete(A[:, :ncols_used], np.arange(ig.start, ig.stop), axis=1)
            A2 = np.hstack([A2[:, :nv], A[:, ig] @ B, A2[:, nv:]])
            b2_rhs = b - A[:, ig] @ (g_mag * g_hat)
            x2, *_ = np.linalg.lstsq(A2, b2_rhs, rcond=None)
            w2 = x2[nv:nv + 2]
            g_new = g_mag * g_hat + B @ w2
            g_hat = g_new / np.linalg.norm(g_new)
            # rebuild full x with the refined g for the final outputs
            x = np.concatenate([x2[:nv], g_mag * g_hat, x2[nv + 2:]])

        res_w = A[:, :len(x)] @ x - b
        res_raw = A0[:, :len(x)] @ x - b0          # unweighted, for reweighting
        return x, g_norm_ls, g_mag * g_hat, svals, estimated_ba, res_w, res_raw

    # pass 1 with a nominal camera-noise floor, then reweight the position
    # rows from their actual (unweighted) residuals — handles clean and noisy
    # pose tracks alike (IRLS, one extra pass)
    x, g_norm_ls, g_refined, svals, estimated_ba, res_w, res_raw = _solve(1e-6)
    cam_pos_var = max(np.mean(res_raw[pos_rows] ** 2) / 2.0, 1e-12)
    x, g_norm_ls, g_refined, svals, estimated_ba, res_w, _ = _solve(cam_pos_var)

    def _sigmas(x, res_w):
        """Parameter 1σ from the (residual-matched) weighted LS — used to GATE
        the weakly-observed quantities (p_IinC, s, b_a): an honest zero / s=1
        beats a wild estimate the joint solve must dig itself out of."""
        w_fin = _weights(cam_pos_var)
        n_used = len(x)
        Aw = (A0 * w_fin[:, None])[:, :n_used]
        dof = max(Aw.shape[0] - n_used, 1)
        scale2 = float(res_w @ res_w) / dof      # residual-variance correction
        try:
            Cdiag = scale2 * np.diag(np.linalg.inv(Aw.T @ Aw))
        except np.linalg.LinAlgError:
            Cdiag = np.full(n_used, np.inf)
        return np.sqrt(np.maximum(Cdiag, 0.0))

    sig = _sigmas(x, res_w)
    if estimated_ba and np.max(sig[iba]) > ba_sigma_gate:
        # b_a too uncertain (also drags gravity with it) — re-solve without it
        x, g_norm_ls, g_refined, svals, estimated_ba, res_w, _ = _solve(
            cam_pos_var, drop_ba=True)
        sig = _sigmas(x, res_w)

    return AlignmentResult(
        s=float(x[i_s]) if estimate_scale else 1.0,
        g_inB=g_refined,
        v_inB=x[:nv].reshape(-1, 3),
        p_IinC=x[ip],
        b_a=x[iba] if estimated_ba else np.zeros(3),
        g_norm_ls=g_norm_ls,
        residual_rms=float(np.sqrt(np.mean(res_w ** 2))),
        svals=svals,
        estimated_ba=estimated_ba,
        t_key=t_key,
        n_skipped_intervals=K - M,
        sigma_p=sig[ip],
        sigma_s=float(sig[i_s]) if estimate_scale else 0.0,
        sigma_ba=sig[iba] if estimated_ba else None)


# ---------------------------------------------------------------------------
# I4 — gravity alignment + seed assembly
# ---------------------------------------------------------------------------
def gravity_align(g_inB, g_mag: float = GRAVITY_MAG) -> np.ndarray:
    """Minimal rotation R_BtoW with R·ĝ_inB = [0,0,−1] (world z-up, yaw-free)."""
    g_hat = np.asarray(g_inB, float)
    g_hat = g_hat / np.linalg.norm(g_hat)
    z_dn = np.array([0.0, 0.0, -1.0])
    axis = np.cross(g_hat, z_dn)
    s, c = np.linalg.norm(axis), float(g_hat @ z_dn)
    if s < 1e-12:
        return np.eye(3) if c > 0 else exp_so3(np.pi * np.array([1.0, 0, 0]))
    return exp_so3(axis / s * np.arctan2(s, c))


@dataclass
class CamImuSeed:
    """Initialization seed for the joint camera-IMU calibration."""
    T_CtoI: gtsam.Pose3
    td: float
    s: float
    b_g: np.ndarray
    b_a: np.ndarray
    g_inB: np.ndarray
    R_BtoW: np.ndarray
    T_BinW: gtsam.Pose3
    t_key: np.ndarray            # keyframe times (camera stamps)
    T_ItoW: list                 # keyframe IMU poses in the gravity world
    v_inW: np.ndarray            # (K+1, 3)
    report: dict = field(default_factory=dict)


def detect_standstill(t_imu, w_m, a_m, window: float = 1.0,
                      w_thresh: float = 0.02, a_thresh: float = 0.1):
    """True if the stream STARTS stationary (variance test on the first window).

    Returns (is_still, b_g_still, mean_accel) — when still, the gyro mean is an
    independent b_g estimate and the accel mean points along gravity-minus-b_a
    (cross-checks for I2/I3, C++ MVIS initIMU-style)."""
    t_imu = np.asarray(t_imu, float)
    m = t_imu <= t_imu[0] + window
    w, a = np.asarray(w_m, float)[m], np.asarray(a_m, float)[m]
    still = (np.std(np.linalg.norm(w, axis=1)) < w_thresh and
             np.std(np.linalg.norm(a, axis=1)) < a_thresh)
    return bool(still), w.mean(0), a.mean(0)


def initialize_cam_imu_from_detections(
        t_cam, ids_list, uv_list, board_points, cal_vec, t_imu, w_m, a_m, *,
        is_fisheye: bool = False, model: str | None = None,
        pnp_method: str = "iterative", **kwargs) -> CamImuSeed:
    """The full cam-IMU init from raw inputs of the separate cam-IMU process:
    board detections + camera intrinsics/distortion (from the camera-only
    calibration) + raw IMU. Runs I0 (PnP) then I1→I4. BOARD MODE (metric)."""
    t_kept, poses, _n = pnp_pose_track(t_cam, ids_list, uv_list,
                                       board_points, cal_vec, is_fisheye,
                                       model=model, pnp_method=pnp_method)
    if len(t_kept) < 10:
        raise ValueError(f"only {len(t_kept)} usable PnP frames")
    return initialize_cam_imu(t_kept, poses, t_imu, w_m, a_m, **kwargs)


def initialize_cam_imu_anonymous(
        t_cam, ids_list, uv_list, cal_vec, t_imu, w_m, a_m, *,
        is_fisheye: bool = False, model: str | None = None,
        sfm_mode: str = "planar",
        sfm_kwargs: dict | None = None, **kwargs) -> CamImuSeed:
    """FEATURE-ONLY MODE: the same init with the board geometry WITHHELD.

    The detections are used purely as anonymous 2-D feature tracks (corner IDs
    = correspondences); the pose track comes from up-to-scale SfM and the IMU
    supplies the metric scale through the scale-unknown alignment (‖g‖ anchors
    the meters). The genuinely-estimated ŝ converts structure units → meters;
    the seed states/T_BinW live in the gravity-aligned STRUCTURE frame (not the
    board frame).

    `sfm_mode` selects the two-view bootstrap:
      - ``"planar"``  — homography (`planar_sfm`); features on one plane/board.
      - ``"general"`` — essential matrix (`general_sfm`); features spread in 3D
        across several boards (e.g. the ArUco-as-features target).
      - ``"auto"``    — try planar; if it fails or reprojects poorly, fall back
        to general. (Cheap heuristic; the standalone planarity probe is the
        authoritative classifier.)"""
    from mvis.calibration.mono_sfm import (
        general_sfm, normalize_detections, planar_sfm)

    tracks = normalize_detections(ids_list, uv_list, cal_vec, is_fisheye,
                                  model=model)
    ids_n = [tr[0] for tr in tracks]
    xy_n = [tr[1] for tr in tracks]
    kw = sfm_kwargs or {}
    if sfm_mode == "planar":
        sfm = planar_sfm(t_cam, ids_n, xy_n, **kw)
    elif sfm_mode == "general":
        sfm = general_sfm(t_cam, ids_n, xy_n, **kw)
    elif sfm_mode == "auto":
        try:
            sfm = planar_sfm(t_cam, ids_n, xy_n, **kw)
            if sfm.reproj_rms > 3.0:        # px — planar bootstrap misfit
                sfm = general_sfm(t_cam, ids_n, xy_n, **kw)
        except ValueError:
            sfm = general_sfm(t_cam, ids_n, xy_n, **kw)
    else:
        raise ValueError(f"unknown sfm_mode: {sfm_mode!r}")
    if len(sfm.t) < 10:
        raise ValueError(f"only {len(sfm.t)} SfM-posed frames")
    seed = initialize_cam_imu(sfm.t, sfm.poses, t_imu, w_m, a_m,
                              board_mode=False, **kwargs)
    seed.report["sfm"] = {
        "mode": sfm_mode,
        "n_frames": len(sfm.t), "n_keyframes": len(sfm.kf_idx),
        "reproj_rms_px": sfm.reproj_rms,
        "planarity_rms": sfm.planarity_rms,
        "bootstrap_parallax": sfm.parallax_med,
    }
    return seed


def initialize_cam_imu(t_cam, T_CtoB_list, t_imu, w_m, a_m, *,
                       board_mode: bool = True,
                       use_imu_scale: bool | None = None,
                       keyframe_hz: float = 2.0, max_offset: float = 0.1,
                       diff_baseline: float = 0.3,
                       estimate_ba: bool = False, g_mag: float = GRAVITY_MAG,
                       g_norm_tol: float = 0.05,
                       scale_rel_tol: float = 0.10) -> CamImuSeed:
    """Run the I1→I4 initialization on a pose track. Default: scale-unknown.

    The alignment always SOLVES the scale s (one code path for board PnP and
    future feature-only tracks). What it does with ŝ depends on the mode:
    `board_mode=True` (metric pose track) USES s = 1 and treats ŝ as the
    consistency gate (`scale_ok` — ŝ far from 1 ⇒ broken intrinsics / units /
    preintegration; note ŝ carries errors-in-variables attenuation under PnP
    noise, so the gate is diagnostic, not an unbiased estimate). Feature-only
    mode (`board_mode=False`) consumes the gated ŝ — beware the EIV bias
    there; an EIV-aware estimator is future work (natural-feature-calib).

    Args:
        t_cam, T_CtoB_list: camera pose track (from `pnp_pose_track`), full rate.
        t_imu, w_m, a_m: raw IMU stream (should extend past the camera window
            by ≥ max_offset on both sides).
        keyframe_hz: decimation rate for the I3 alignment. Keep LOW (~1-2 Hz):
            the keyframe position differences must dominate the PnP position
            noise, else errors-in-variables attenuation biases the scale
            (σ_s cannot see that bias — only longer spacing prevents it).
        diff_baseline: camera-ω differentiation baseline (s) for the td /
            rotation stages — long enough to average down PnP noise.
        g_norm_tol: relative gate on ‖g_LS‖ vs g_mag (the primary quality gate).
    """
    t_cam = np.asarray(t_cam, float)
    R_list = [T.rotation().matrix() for T in T_CtoB_list]
    stride = _stride_for_baseline(t_cam, diff_baseline)

    # I1 — time offset: coarse correlation (global) + LS-valley refinement
    td0, rep_td = estimate_time_offset_coarse(t_cam, R_list, t_imu, w_m, max_offset)
    td, rep_refine, pre = refine_time_offset(t_cam, R_list, t_imu, w_m, td0,
                                             stride=stride, return_precomp=True)
    rep_td["td_coarse"] = td0
    rep_td["td"] = td
    rep_td.update(rep_refine)

    # I2 — rotation + gyro bias (at the refined td); the camera ω / gyro
    # dead-reckoning of I1 is reused (bitwise: same inputs, same stride)
    R_CtoI, b_g, rep_rot = estimate_rotation_gyro(t_cam, R_list, t_imu, w_m, td,
                                                  stride=stride, _precomp=pre)

    # I3 — alignment on decimated keyframes; IMU intervals sampled at t + td
    kf_stride = max(1, int(round((len(t_cam) - 1)
                                 / ((t_cam[-1] - t_cam[0]) * keyframe_hz))))
    idx = np.arange(0, len(t_cam), kf_stride)
    t_key_all = t_cam[idx] + td                    # keyframes on the IMU clock
    T_key = [T_CtoB_list[i] for i in idx]
    align = linear_alignment(t_key_all, T_key, R_CtoI, b_g, t_imu, w_m, a_m,
                             estimate_scale=True, estimate_ba=estimate_ba,
                             g_mag=g_mag)
    # map back to the keyframes the alignment actually kept (gap filtering)
    kept_pos = np.searchsorted(t_key_all, align.t_key)
    idx = idx[kept_pos]
    t_key = align.t_key

    # gate weakly-observed quantities (honest fallbacks beat wild estimates;
    # the joint solve refines them from there)
    p_gated = bool(np.max(align.sigma_p) > 0.05)       # extrinsic translation
    # the scale gate is RELATIVE — in feature mode the structure scale (and
    # hence ŝ) is arbitrary-magnitude
    s_rel_sigma = align.sigma_s / max(abs(align.s), 1e-9)
    s_gated = bool(s_rel_sigma > 0.02)
    p_IinC = np.zeros(3) if p_gated else align.p_IinC
    # Scale handling is INDEPENDENT of board_mode: the
    # alignment always solves ŝ; `use_imu_scale` chooses whether to USE it.
    #   - use_imu_scale=False -> s=1 (trust known metric geometry; ŝ is the
    #     consistency gate). Only valid with a metric pose track (board PnP).
    #   - use_imu_scale=True  -> s=ŝ (scale recovered from the IMU; no scale
    #     assumption). Works with the board PnP track too — useful to validate
    #     ŝ against the known board (ŝ should ≈ 1) free of SfM noise.
    # Default mirrors the old behaviour: board_mode -> known scale.
    if use_imu_scale is None:
        use_imu_scale = not board_mode
    if use_imu_scale:
        # NEEDS ŝ — abort only when it is truly unobservable (a seed tolerates
        # ~10-20% scale error; the joint solve refines it). `scale_rel_tol` can
        # be loosened when scale is weak but a SCALE-ROBUST quantity is the
        # target (e.g. rolling-shutter tr / td on a calm bag).
        if s_rel_sigma > scale_rel_tol:
            raise ValueError(
                f"scale unobservable (σ_s/ŝ = {s_rel_sigma:.3f} > "
                f"{scale_rel_tol}) — insufficient acceleration excitation for "
                "scale-from-IMU init")
        s_used = align.s
    else:
        s_used = 1.0          # metric track: ŝ is the consistency gate only

    # I4 — gravity-aligned world + assembly
    R_BtoW = gravity_align(align.g_inB, g_mag)
    p_CinI = -R_CtoI @ p_IinC
    T_CtoI = gtsam.Pose3(gtsam.Rot3(R_CtoI), gtsam.Point3(*p_CinI))
    T_BinW = gtsam.Pose3(gtsam.Rot3(R_BtoW), gtsam.Point3(0, 0, 0))

    T_ItoW, v_inW = [], []
    for i, k in enumerate(idx):
        R_IB_k = np.asarray(T_CtoB_list[k].rotation().matrix(), float) @ R_CtoI.T
        p_IB_k = (s_used * np.asarray(T_CtoB_list[k].translation(), float)
                  + np.asarray(T_CtoB_list[k].rotation().matrix(), float) @ p_IinC)
        T_ItoW.append(gtsam.Pose3(gtsam.Rot3(R_BtoW @ R_IB_k),
                                  gtsam.Point3(*(R_BtoW @ p_IB_k))))
        v_inW.append(R_BtoW @ align.v_inB[i])

    still, bg_still, _acc_still = detect_standstill(t_imu, w_m, a_m)
    g_norm_ok = abs(align.g_norm_ls - g_mag) / g_mag < g_norm_tol
    report = {
        "td": rep_td, "rotation": rep_rot,
        "scale": align.s,
        "g_norm_ls": align.g_norm_ls,
        "g_norm_ok": bool(g_norm_ok),              # PRIMARY quality gate
        # board mode: ŝ carries ~2% errors-in-variables attenuation under real
        # PnP noise (verified in sim AND on the VI-rig) — gate at 3%.
        # feature mode: ŝ is arbitrary-magnitude — the ŝ≈1 check is meaningless
        "scale_ok": bool(abs(align.s - 1.0) < 0.03) if board_mode else None,
        "alignment_residual_rms": align.residual_rms,
        "estimated_ba": align.estimated_ba,
        "p_gated": p_gated, "sigma_p": align.sigma_p,
        "s_gated": s_gated, "sigma_s": align.sigma_s,
        "n_skipped_intervals": align.n_skipped_intervals,
        "standstill": still,
        "bg_standstill_agree": (float(np.linalg.norm(bg_still - b_g))
                                if still else None),
    }
    return CamImuSeed(
        T_CtoI=T_CtoI, td=td, s=s_used, b_g=b_g, b_a=align.b_a,
        g_inB=align.g_inB, R_BtoW=R_BtoW, T_BinW=T_BinW,
        t_key=t_key, T_ItoW=T_ItoW, v_inW=np.array(v_inW), report=report)


# ---------------------------------------------------------------------------
# I6 — IMU-IMU initialization (multi-IMU). Direct gyro-to-gyro: both
# sides are DIRECT angular-rate measurements at 100-500 Hz, so td and the
# relative rotation come out far sharper than chaining through the camera
# (whose omega is differentiated 28 Hz PnP — the noisiest stage above).
# Translation (lever arm) is NOT initialized (0): it is only observable
# through angular-acceleration terms the joint solve owns.
# ---------------------------------------------------------------------------
@dataclass
class ImuImuSeed:
    """Init seed for one auxiliary IMU relative to the base IMU.

    Convention (matches `ImuCalib`/`simulate_imu`): an aux sample stamped t
    is the physical state at base-clock t + td.
    """
    T_IstoI: gtsam.Pose3         # rotation from Kabsch; translation = 0
    td: float
    b_g_s: np.ndarray            # aux gyro bias (corrected by b_g_base if given)
    report: dict = field(default_factory=dict)


def _boxcar(w, t, width):
    """Same-kernel moving average per axis. Applied to BOTH streams it adds
    no relative bias and cuts white noise by sqrt(n) — the interval-mean
    lesson from I1/I2, here as a pre-filter for the gyro-gyro pairing."""
    if width <= 0:
        return np.asarray(w, float)
    dt = float(np.median(np.diff(t)))
    n = max(int(round(width / dt)), 1)
    if n % 2 == 0:
        n += 1      # ODD kernel: even-length 'same' convolution shifts the
                    # signal by half a sample — a td bias when rates differ
    if n < 3:
        return np.asarray(w, float)
    k = np.ones(n) / n
    w = np.asarray(w, float)
    return np.stack([np.convolve(w[:, i], k, mode='same')
                     for i in range(w.shape[1])], axis=1)


def _gyro_pair(t_base, w_base, t_aux, w_aux, td, sample_hz):
    """Paired (base, aux) angular-rate samples on a uniform base-clock grid:
    base interpolated at tau, aux at tau - td (aux stamp of base time tau)."""
    lo = max(t_base[0], t_aux[0] + td) + 0.05
    hi = min(t_base[-1], t_aux[-1] + td) - 0.05
    tau = np.arange(lo, hi, 1.0 / sample_hz)
    wb = _interp_vec(tau, t_base, w_base)
    wa = _interp_vec(tau - td, t_aux, w_aux)
    return wb, wa


def _kabsch_rms(wb, wa):
    """Centered Kabsch R (aux->base) + residual RMS [rad/s]."""
    cb, ca = wb - wb.mean(0), wa - wa.mean(0)
    U, _S, Vt = np.linalg.svd(cb.T @ ca)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    rms = float(np.sqrt(np.mean(np.sum((cb - ca @ R.T) ** 2, axis=1))))
    return R, rms


def initialize_imu_imu(t_base, w_base, t_aux, w_aux, *,
                       b_g_base=None,
                       max_offset: float = 0.1, sample_hz: float = 100.0,
                       smooth_s: float = 0.025,
                       valley_halfwidth: float = 0.05) -> ImuImuSeed:
    """I6: time offset + relative rotation + aux gyro bias, gyro-to-gyro.

    1. coarse td — cross-correlation of detrended |omega| on a uniform grid
    2. refine    — LS valley: centered-Kabsch residual RMS over a td grid,
                   two narrowing passes + parabola (same recipe as I1);
                   both streams boxcar-smoothed (`smooth_s`, same kernel —
                   noise / sqrt(n), no relative bias)
    3. R_IstoI   — Kabsch at the refined td; b_g_s from the paired means,
                   corrected by `b_g_base` (the base seed's gyro bias) when
                   given — otherwise b_g_s absorbs R^T b_g_base.

    Gates in report: valley_ratio (rms at +-valley_halfwidth / rms_min),
    kabsch_rms, corr_peak_ratio, overlap_s.
    """
    t_base = np.asarray(t_base, float)
    t_aux = np.asarray(t_aux, float)
    w_base_raw = np.asarray(w_base, float)
    w_base = _boxcar(w_base_raw, t_base, smooth_s)
    w_aux = _boxcar(np.asarray(w_aux, float), t_aux, smooth_s)

    # --- 1. coarse correlation of |omega| (detrended by a moving mean) ---
    grid_hz = 200.0
    lo = max(t_base[0], t_aux[0]) + 0.02
    hi = min(t_base[-1], t_aux[-1]) - 0.02
    tau = np.arange(lo, hi, 1.0 / grid_hz)
    nb = np.linalg.norm(_interp_vec(tau, t_base, w_base), axis=1)
    na = np.linalg.norm(_interp_vec(tau, t_aux, w_aux), axis=1)
    win = max(int(grid_hz * 1.0), 1)
    kern = np.ones(win) / win
    nb = nb - np.convolve(nb, kern, mode='same')
    na = na - np.convolve(na, kern, mode='same')
    max_lag = int(max_offset * grid_hz)
    corr = np.correlate(nb, na, mode='full')
    mid = len(na) - 1
    sl = corr[mid - max_lag: mid + max_lag + 1]
    lags = np.arange(-max_lag, max_lag + 1)
    # base(tau) ~ aux(tau - td): |omega|_aux shifted by +td aligns with base
    td_coarse = float(lags[np.argmax(sl)] / grid_hz)
    corr_peak_ratio = float(np.max(sl) / max(np.median(np.abs(sl)), 1e-12))

    # --- 2. LS-valley refinement (Kabsch residual over a td grid) ---
    td = td_coarse
    span = 0.004
    for _pass in range(2):
        grid = np.arange(td - span, td + span + 1e-12, span / 8)
        rms = np.array([_kabsch_rms(*_gyro_pair(
            t_base, w_base, t_aux, w_aux, g, sample_hz))[1] for g in grid])
        k = int(np.argmin(rms))
        td = float(grid[k])
        if 0 < k < len(grid) - 1:               # parabola through the valley
            d1 = rms[k + 1] - rms[k - 1]
            d2 = rms[k + 1] - 2 * rms[k] + rms[k - 1]
            if d2 > 0:
                td -= (grid[1] - grid[0]) * 0.5 * d1 / d2
        span /= 8

    # --- 3. rotation + bias at the refined td, plus gates ---
    wb, wa = _gyro_pair(t_base, w_base, t_aux, w_aux, td, sample_hz)
    R, rms_min = _kabsch_rms(wb, wa)
    # w_aux = R^T (w_base - b_g_base) + b_g_s  =>
    bgb = np.zeros(3) if b_g_base is None else np.asarray(b_g_base, float)
    b_g_s = np.asarray(wa.mean(0) - R.T @ (wb.mean(0) - bgb), float)
    rms_off = np.mean([_kabsch_rms(*_gyro_pair(
        t_base, w_base, t_aux, w_aux, td + s * valley_halfwidth,
        sample_hz))[1] for s in (-1.0, 1.0)])
    report = {
        "td_coarse": td_coarse, "td": td,
        "kabsch_rms": rms_min,
        "valley_ratio": float(rms_off / max(rms_min, 1e-12)),
        "corr_peak_ratio": corr_peak_ratio,
        "overlap_s": float(min(t_base[-1], t_aux[-1])
                           - max(t_base[0], t_aux[0])),
        "n_pairs": int(len(wb)),
    }
    report["ok"] = bool(report["valley_ratio"] > 1.5
                        and report["overlap_s"] > 10.0)
    return ImuImuSeed(
        T_IstoI=gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(0.0, 0.0, 0.0)),
        td=td, b_g_s=b_g_s, report=report)
