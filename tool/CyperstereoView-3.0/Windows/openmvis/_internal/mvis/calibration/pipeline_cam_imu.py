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
"""Joint single-camera + single-IMU calibration pipeline.

The production assembly around the graph proven in `test_cam_imu_recovery`:

  1. world frame  — from the init's gravity alignment (g_inW = [0,0,-G] FIXED)
  2. seed         — cam_imu_init over the WHOLE sequence (velocities for all
                    alignment keyframes; b_g from Kabsch, b_a = 0)
  3. states       — (X_k = T_ItoW, vb_k = [v, b_g, b_a]) at every accepted
                    camera frame stamp (optionally decimated by `state_hz`);
                    poses from the PnP track, velocities densified from the
                    seed keyframes by ACI propagation
  4. graph        — ImuFactorACI_In per interval (15-dim, bias walk included)
                    + CameraCalibImuFactor / CameraCalibInterpFactor per frame
                    (Huber) + 4-DOF gauge on
                    X(0) (position + yaw about gravity; roll/pitch FREE — they
                    are observable, anchoring the seed's tilt error would leak
                    into b_a/Da) + vb(0) prior + IMU-intrinsics prior (phase A
                    pins ξ_in = 0) + tr pin (global shutter)
  5. optimize     — batch LM, analytic Jacobians throughout
  6. (extension beyond init coverage: a fallback — whole-bag init is
     the default, so every state is covered)
  7. diagnostics  — per-frame reprojection RMS pre/post, IMU whitened cost
                    split by rows, chi^2/DOF per factor type, init-vs-joint
                    parameter shifts, calib marginals
  8. report       — CameraCalib (intrinsics, T_CtoI, td, tr) + ImuCalib
                    (biases; Dw/Da/Tg pinned in phase A) with sigmas

Detection gaps are BRIDGED by a single preintegrated factor regardless of
length (no segment splitting — future multi-camera rigs constrain spans where
the base camera is blind); watch `imu_interval_cost` for long-gap pathology.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import gtsam

from .cam_imu_init import (CamImuSeed, initialize_cam_imu,
                           initialize_imu_imu, pnp_pose_track, _slice_imu,
                           _interp_vec, _is_monotonic)
from .factors.aci import (ACIPreintegrator, ACIGyroPreintegrator, _IntrPack,
                          correct_measurements, state_step)
from .factors.camera_calib import CameraCalib
from mvis.detection.corner_subsample import corner_mask
from .board_shape import board_height_gauge_weights
from .factors.camera_interp_factor import CameraCalibInterpFactor
from .factors.pose_interpolation import interp_pose
from .factors.camera_imu_factor import CameraCalibImuFactor
from .factors.imu_segment import ImuSegment
from mvis.core.shutter import row_offset
from mvis.core.gravity import describe as describe_gravity
from .factors.camera_feature_factor import CameraFeatureInterpFactor
from .factors.imu_factor import ImuFactorACI_In
from .factors.imu_factor_multi import ImuFactorACI_Multi_In_V
from .factors.imu_factor_gyro import ImuFactorACI_Multi_In_SO3
from mvis.imu import (
    create_imu_model, RPNG, parse_imu_model, imu_model_name,
    intrinsic_sigma_vector, intrinsic_block_slices,
)
from .factors.lie_groups import log_so3, so3_right_jacobian_inv
from .reproj_stats import reproj_entry
from .state_dump import maybe_perturb_initial_values, maybe_dump_state
from .lm_params import (LM_LAMBDA_LOWER_DEFAULT, LM_LAMBDA_UPPER_DEFAULT,
                        make_lm_params)
# The solve-knobs config lives canonically in core.config; re-exported here so
# existing `from ...pipeline_cam_imu import CamImuCalibConfig` callers keep working.
from mvis.core.config import CamImuCalibConfig
from mvis.calibration.lm_progress import optimize_logged
from mvis.calibration.lm_rescue import rescue_if_stuck
from mvis.calibration.tga_relin import TgaRelinOptimizer, TgaStream
from mvis.imu.resample import upsample_imu
from mvis.calibration.corner_accounting import CornerTally, guarded, not_in_solve, warn_past_90
from mvis.core.camera_models import KB_THETA_MODELS, dist_names, legacy_model

logger = logging.getLogger(__name__)


def _note_kb_domain(entry: dict, c: int) -> None:
    """kb4 only: log the fit's valid domain (first theta_d fold vs the image
    corner; WARNING when it folds inside the image) and keep it on the camera
    entry as `valid_domain` for the reports. Never fails the run."""
    cal = entry['calib']
    if cal.model in KB_THETA_MODELS:
        from mvis.camera.kb_domain import check_kb_valid_domain
        entry['valid_domain'] = check_kb_valid_domain(
            cal.intr, entry.get('image_size'), cal.model, label=f"cam{c}", log=logger)
    n_past = (entry.get('corner_accounting') or {}).get('past_fold', 0)
    if n_past:
        logger.warning(f"cam{c}: ALARM: {n_past} observed corners lie past the "
                       f"{cal.model} fold; they have zero residual, so they lower "
                       f"the cost instead of raising it")
    warn_past_90(entry.get('corner_accounting'), f"cam{c}", logger)


def _frame_fate(cam, considered, used) -> dict:
    """Frames and corners of `cam` that reached no vision factor. `considered`: the frame times the
    solve tried, so after rate decimation and including those the init rejected (PnP); `used`: the
    times that got a factor. A frame decimated away is not a no-pose frame, it is `not_in_solve`."""
    ids = cam['ids_list']
    t_cam = [float(t) for t in cam['t_cam']]
    considered, used = set(float(t) for t in considered), set(float(t) for t in used)
    lost = [i for i, t in enumerate(t_cam) if t in considered and t not in used]
    return {'no_pose_frames': len(lost), 'no_pose_corners': int(sum(len(ids[i]) for i in lost)),
            'detected': int(sum(len(x) for x in ids))}


def _corner_accounting(cam_factors, values, sols, image_sizes, fates, outliers=None) -> list:
    """Per camera, the `corner_accounting` count of the vision factors at the final `values` (one
    dict per camera). `fates` per camera from `_frame_fate`, `outliers` per camera where the path
    rejects corners. Reads only."""
    tallies = [CornerTally(c, s.model, s.intr) for c, s in enumerate(sols)]
    for c, j, fac in cam_factors:
        pts = fac.camera_points_values(values)
        norms = _frame_corner_norms(fac, c, j, values)
        for p, e in zip(pts, norms):
            tallies[c].add(p, bool(np.isfinite(e)))
    out = []
    for c, t in enumerate(tallies):
        t.no_pose_frames = fates[c]['no_pose_frames']
        t.no_pose_corners = fates[c]['no_pose_corners']
        t.outliers = int(outliers[c]) if outliers is not None else 0
        a = t.to_dict(image_sizes[c])
        a['detected'] = fates[c]['detected']
        a['not_in_solve'] = not_in_solve(a)
        out.append(a)
    return out


def _guarded_accounting(n_cams, count) -> list:
    """`count()`, the per-camera list of `_corner_accounting`, or None per camera if it fails."""
    return guarded(count) or [None] * n_cams


def _imu_model(cfg) -> int:
    """Resolve the configured IMU-intrinsic model id (defaults to CALYX)."""
    return parse_imu_model(getattr(cfg, "imu_intrinsics_model", "calyx"))


def _cam_model(cam: dict) -> str:
    """Resolve a per-camera dict's distortion model (canonical id), tolerating
    the legacy `is_fisheye`-only spec."""
    m = cam.get('model')
    if m:
        return str(m)
    return legacy_model(bool(cam.get('is_fisheye', False)), cam.get('cal_vec'))


X = gtsam.symbol_shorthand.X
_Vk = lambda k: gtsam.symbol('v', k)
_B = lambda b: gtsam.symbol('b', int(b))          # per-board pose (b=0 is the anchor)
B0 = _B(0)                                         # single-board alias (backward compat)
_L = lambda i: gtsam.symbol('l', i)               # feature-mode world landmark


def board_poses_from_values(values, board_ids):
    """Extract {board_id: gtsam.Pose3} for the given board IDs from solved Values."""
    return {int(b): values.atPose3(_B(int(b))) for b in board_ids}
_Cc = lambda c: gtsam.symbol('c', c)
C0 = _Cc(0)
I0 = gtsam.symbol('I', 0)
O0 = gtsam.symbol('o', 0)                          # base-IMU gyro<->accel t_ga (Vector1)
_Fdt = lambda n: gtsam.symbol('f', n)              # per-frame capture-time offset (Vector1)
H0 = gtsam.symbol('H', 0)                          # board shape: one height per board point (VectorN)


def _board_heights_report(res, marg, board_points, pids_per_cam) -> dict:
    """Solved board shape: total height per board point (the offsets H0 plus
    the heights the board points started with) and its sigma, the number of
    vision observations per point, and summary stats over the observed points."""
    dh = np.asarray(res.atVector(H0), float)
    h0 = np.asarray(board_points, float)[:, 2]
    h = h0 + dh
    n_obs = np.zeros(len(dh), int)
    for frames in pids_per_cam:
        for pids in frames:
            np.add.at(n_obs, np.asarray(pids, int), 1)
    sig = None
    if marg is not None:
        try:
            sig = np.sqrt(np.clip(np.diag(marg.marginalCovariance(H0)), 0, None))
        except Exception:                      # noqa: BLE001 — sigma is optional
            sig = None
    seen = n_obs > 0
    return {'heights_m': h, 'offset_m': dh, 'initial_m': h0, 'sigma_m': sig,
            'n_obs': n_obs,
            'rms_mm': float(np.std(h[seen]) * 1e3) if seen.any() else None,
            'p2p_mm': float(np.ptp(h[seen]) * 1e3) if seen.any() else None,
            'offset_rms_mm': float(np.sqrt(np.mean(dh[seen] ** 2)) * 1e3) if seen.any() else None,
            'n_seen': int(seen.sum())}


def _board_height_factors(pts_B, sigma_prior, sigma_gauge=1e-6):
    """Weak prior + gauge on the board-height offsets h (key H0).

    The offsets are added along the board normal to the board points the graph
    already uses (flat, or board.heights_file). The gauge pins their mean and
    first moments about the board x / y axes (each row in metres) so the board
    pose keeps the mean plane and tilt — the same split as the camera-only
    solver's `_board_height_gauge_factor`. Corners never observed stay at the
    prior."""
    pts_B = np.asarray(pts_B, float)
    N = len(pts_B)
    W = board_height_gauge_weights(pts_B)
    noise = gtsam.noiseModel.Isotropic.Sigma(3, sigma_gauge)

    def err(this, values, H):
        h = values.atVector(H0)
        if H is not None:
            H[0] = W.T.copy()
        return W.T @ h

    gauge = gtsam.CustomFactor(noise, gtsam.KeyVector([H0]), err)
    prior = gtsam.PriorFactorVector(H0, np.zeros(N),
                                    gtsam.noiseModel.Isotropic.Sigma(N, sigma_prior))
    return [gauge, prior]


def _lm_no_step_report(graph, values, opt, cfg):
    """None if LM moved. At 0 iterations, tell a real failure from a start LM could not improve.

    GTSAM swallows a failed linear solve: when Cholesky fails at every lambda, LM stops with 0
    iterations and err_final == err_init, like a start already at the minimum. One linearize + solve
    at the start (only here, so a solve that iterated does no extra work) tells them apart:
      failed True:  the linear system cannot be solved, the result is just the initial values;
      failed False: it can, and LM found no better point. `predicted_decrease` is what the
                    undamped step promised: about 0 at a true minimum."""
    if opt.iterations() > 0 or int(cfg.max_iterations) <= 0:
        return None
    rep = {'failed': False, 'err': float(opt.error()), 'predicted_decrease': None}
    try:
        gfg = graph.linearize(values)
        delta = gfg.optimize()
        rep['predicted_decrease'] = float(gfg.error(delta.scale(0.0)) - gfg.error(delta))
        if not np.isfinite(rep['predicted_decrease']):
            rep.update(failed=True, reason="the linear solve is not finite")
    except RuntimeError as e:                  # gtsam's IndeterminantLinearSystemException
        rep.update(failed=True, reason=" ".join(str(e).split())[:200])
    if rep['failed']:
        logger.warning("LM made no step: the linear solve fails (%s). The result is the initial "
                       "values; marking the solve not ok. Look for tiny-sigma priors or "
                       "unconstrained states.", rep['reason'])
    else:
        rep['reason'] = "no better point than the start"
        logger.warning("LM made no step: the linear solve works, but the start could not be improved "
                       "(error %.6g, predicted decrease %.3g).", rep['err'], rep['predicted_decrease'])
    return rep


def _lm_params(cfg, lambda_initial=None):
    """LM params for the joint solve: `cfg.max_iterations`, GTSAM default
    tolerances (1e-5) and the `cfg.lm_lambda_lower/upper` damping bounds
    (cam-imu-speedup L1; `make_lm_params` has the semantics)."""
    return make_lm_params(
        cfg.max_iterations,
        lambda_lower=getattr(cfg, "lm_lambda_lower", LM_LAMBDA_LOWER_DEFAULT),
        lambda_upper=getattr(cfg, "lm_lambda_upper", LM_LAMBDA_UPPER_DEFAULT),
        lambda_initial=lambda_initial)


def _Os(s):
    # Aux-IMU s's OWN gyro<->accel t_ga (Vector1); base takes 'o' index 0.
    return gtsam.symbol('o', 1 + s)
_As = lambda s: gtsam.symbol('A', s)              # aux ImuCalib xi31
_VBs = lambda s, k: gtsam.symbol('u', s * 1_000_000 + k)  # aux [bg,ba,v]
_BGs = lambda s, k: gtsam.symbol('g', s * 1_000_000 + k)  # gyro-only aux bg(3)

_BIG = 1e3          # "free" sigma for partially-pinned vector priors
_TINY = 1e-8

# valid-domain bounds per model (solve.bound_intrinsics): (index into intr, lo, hi)
_INTR_BOUNDS = {
    'double_sphere': [(0, 0.0, None), (1, 0.0, None), (4, -1.0, 1.0), (5, 0.0, 1.0)],
    'omni_radtan': [(0, 0.0, None), (1, 0.0, None), (4, 0.0, None)],
    'kb4': [(0, 0.0, None), (1, 0.0, None)],   # also 'fisheye' / 'kannala_brandt' (resolve to kb4)
}
_BOUND_SIGMA = 1e-4


def intr_bound_factor(key, base):
    """One-sided hinge on a camera bundle's intrinsics: residual 0 inside the
    model's valid domain (`_INTR_BOUNDS`), (bound - v)/sigma below a lower bound
    and (v - bound)/sigma above an upper one. None for an unbounded model."""
    rows = [(j, b, s) for j, lo, hi in _INTR_BOUNDS.get(base.model, [])
            for b, s in ((lo, -1.0), (hi, 1.0)) if b is not None]
    if not rows:
        return None
    dim = base.DIM

    def err_func(this, values, H):
        xi = values.atVector(key)
        e = np.zeros(len(rows))
        J = np.zeros((len(rows), dim))
        for r, (j, b, s) in enumerate(rows):
            v = base.intr[j] + xi[8 + j]
            if s * (v - b) > 0:
                e[r] = s * (v - b)
                J[r, 8 + j] = s
        if H is not None:
            H[0] = J
        return e

    return gtsam.CustomFactor(gtsam.noiseModel.Isotropic.Sigma(len(rows), _BOUND_SIGMA),
                              gtsam.KeyVector([key]), err_func)


# ---------------------------------------------------------------------------
# CamImuCalibConfig is imported above from mvis.core.config — the
# solve-knobs schema lives there so one config system drives the whole run.


@dataclass
class CamImuCalibResult:
    calib: CameraCalib              # solved BASE bundle (T_CtoI, td, tr, intr)
    calib_sigma: np.ndarray         # (8 + n_intr,) marginal sigmas on base xi
    b_g: np.ndarray                 # bias estimate at the first state
    b_a: np.ndarray
    b_g_drift: np.ndarray           # last-state minus first-state bias
    b_a_drift: np.ndarray
    imu_intr_xi: np.ndarray         # (24,) ξ_in (zeros while pinned)
    imu_intr_sigma: np.ndarray | None
    T_BinW: gtsam.Pose3
    seed: CamImuSeed
    values: gtsam.Values            # full converged state
    t_states: np.ndarray
    diagnostics: dict
    ok: bool
    imu_model: str = "rpng"         # IMU-intrinsic model name (kalibr|rpng|calyx)
    # per-camera entries (index 0 = base): dicts with keys calib,
    # calib_sigma, reproj_rms_px, n_frames, n_dropped, and for extras
    # T_CtoC0 (+ T_CtoC0_sigma approx), seed_td, seed_gates
    cameras: list = field(default_factory=list)
    # per-extra-IMU entries: dicts with T_IstoI, td, b_g_s, b_a_s,
    # calib_sigma (31), chi2_dof, seed_report
    imus: list = field(default_factory=list)
    # 2c-12: per-cam {timestamp: "stage:reason"} for every frame the solve
    # did NOT use; the attach folds these into the capture's frame flags
    frame_selection: list = field(default_factory=list)


# ---------------------------------------------------------------------------
class GaugeFactorPose3:
    """4-DOF gauge prior on one Pose3: position (3) + yaw about WORLD z (1).

    Residual [e_z . Log(R R0^T); p - p0]. Roll/pitch are deliberately
    unconstrained — with gravity fixed in the world model they are observable
    from the accelerometer, and the seed's tilt carries error that the solve
    must be free to correct. First-order Jacobian (exact at the gauge point);
    a gauge factor only needs to remove the 4 datum null-directions.
    """

    def __init__(self, T0: gtsam.Pose3, sigma: float = 1e-6):
        self.R0 = T0.rotation().matrix()
        self.p0 = np.asarray(T0.translation(), float)
        self.sigma = sigma

    def error(self, pose: gtsam.Pose3):
        R = pose.rotation().matrix()
        r = np.empty(4)
        r[0] = log_so3(R @ self.R0.T)[2]
        r[1:4] = np.asarray(pose.translation(), float) - self.p0
        return r

    def error_and_jacobian(self, pose: gtsam.Pose3):
        R = pose.rotation().matrix()
        H = np.zeros((4, 6))
        # R Exp(dth) R0^T = (R R0^T) Exp(R0 dth)  =>
        # d Log/d(dth) = Jr^-1(theta_w) R0,  theta_w = Log(R R0^T)
        th_w = log_so3(R @ self.R0.T)
        H[0, 0:3] = (so3_right_jacobian_inv(th_w) @ self.R0)[2, :]
        H[1:4, 3:6] = R                       # d p / d(delta_rho) = R
        return self.error(pose), H

    def make_factor(self, k_pose):
        noise = gtsam.noiseModel.Isotropic.Sigma(4, self.sigma)

        def err_func(this, values, H):
            pose = values.atPose3(k_pose)
            if H is None:
                return self.error(pose)
            e, J = self.error_and_jacobian(pose)
            H[0] = J
            return e

        return gtsam.CustomFactor(noise, gtsam.KeyVector([k_pose]), err_func)


# ---------------------------------------------------------------------------

def _propagate_state(T0: gtsam.Pose3, v0, t0, t1, t_imu, w_m, a_m, b_g, b_a,
                     cfg: CamImuCalibConfig, g_inW):
    """ACI-propagate (T0, v0) at t0 to t1 (identity intrinsics).

    Only `D_R / D_v / D_p / Dt` are read, so the intrinsic / t_ga Jacobian
    accumulation is skipped (bitwise on those)."""
    pre = ACIPreintegrator(cfg.sigma_g, cfg.sigma_bg, cfg.sigma_a, cfg.sigma_ba,
                           model=_imu_model(cfg), avg_meas=True,
                           track_intrinsics=False)
    pre.reset(np.asarray(b_g, float), np.asarray(b_a, float))
    pre.set_intrinsics(create_imu_model(_imu_model(cfg)))
    t, w, a = _slice_imu(t_imu, w_m, a_m, t0, t1)
    pre.feed_imu_interval(t, w, a)
    R0 = T0.rotation().matrix()
    p0 = np.asarray(T0.translation(), float)
    dt = pre.Dt
    R1 = R0 @ pre.D_R
    v1 = v0 + g_inW * dt + R0 @ pre.D_v
    p1 = p0 + v0 * dt + 0.5 * g_inW * dt * dt + R0 @ pre.D_p
    return gtsam.Pose3(gtsam.Rot3(R1), gtsam.Point3(*p1)), v1


def _densify_velocities(t_states, T_states, seed: CamImuSeed, t_imu, w_m, a_m,
                        cfg: CamImuCalibConfig, g_inW):
    """Per-state velocity init: ACI propagation from the nearest seed keyframe
    at or before each state (constant extrapolation outside seed coverage).

    A speed-up — one shared prefix per seed keyframe instead of one full
    preintegration per state. Per state i the earlier code fed the sample
    sequence [interp(t_key[j]), raw samples strictly inside (t_key[j], t_i),
    interp(t_i)] through `ACIPreintegrator.feed_imu_interval`; the state
    after each raw sample depends only on the samples so far, so the states
    of one keyframe share that walk. The IMU is walked once from `t_key[j]`
    (the interpolated first sample, then every raw sample strictly inside up
    to the group's last state), taking a COPIED `(D_R, D_v, D_p, Dt)`
    snapshot after each raw sample. A state takes the snapshot after the last
    raw sample strictly before `t_i` — a state exactly on a raw sample uses
    the snapshot before that sample, because the strict inequality leaves the
    sample out of its slice — then one `state_step` to the interpolated
    sample at `t_i`, and `v1 = v0 + g·Dt + R0 @ D_v` as before. Measurement
    correction / averaging and the step are `aci.correct_measurements` /
    `aci.state_step`, the helpers `feed_imu_single` itself uses, so the walk
    is bitwise `_propagate_state` (asserted against the frozen copy in
    `tests/_reference/ref_diagnostics.py`). The `j < 0` and `< 1e-9` branches
    are unchanged. A non-monotonic IMU stream (bag stamps out of order) keeps
    the per-state path, whose mask slice does not need sorted stamps.
    """
    t_key = np.asarray(seed.t_key, float)
    v_key = np.asarray(seed.v_inW, float)
    t_states = np.asarray(t_states, float)
    t_imu = np.asarray(t_imu, float)
    v = np.empty((len(t_states), 3))
    shared = _is_monotonic(t_imu)
    groups: dict[int, list[int]] = {}
    for i, t in enumerate(t_states):
        j = int(np.searchsorted(t_key, t, side='right')) - 1
        if j < 0:
            v[i] = v_key[0]
            continue
        if t - t_key[j] < 1e-9:
            v[i] = v_key[j]
            continue
        if not shared:
            # propagate v only (pose comes from PnP); cap at the next keyframe
            _T, v1 = _propagate_state(
                seed.T_ItoW[j], v_key[j], t_key[j], t, t_imu, w_m, a_m,
                seed.b_g, seed.b_a, cfg, g_inW)
            v[i] = v1
            continue
        groups.setdefault(j, []).append(i)
    if not groups:
        return v

    w_m = np.asarray(w_m, float)
    a_m = np.asarray(a_m, float)
    b_g = np.asarray(seed.b_g, float).copy()
    b_a = np.asarray(seed.b_a, float).copy()
    pack = _IntrPack(create_imu_model(_imu_model(cfg)))     # identity intrinsics
    for j, idx in groups.items():
        t0 = float(t_key[j])
        R0 = seed.T_ItoW[j].rotation().matrix()
        v0 = v_key[j]
        ts = t_states[idx]
        # raw samples strictly inside (t0, max t_i): the union of every
        # state's strictly-inside slice (same index rule as `_slice_imu`)
        lo = int(np.searchsorted(t_imu, t0, side='right'))
        hi = max(int(np.searchsorted(t_imu, float(ts.max()), side='left')), lo)
        w0 = _interp_vec([t0], t_imu, w_m)[0]
        a0 = _interp_vec([t0], t_imu, a_m)[0]
        # prefix walk: snaps[m] = state after the m-th raw sample (snaps[0] =
        # identity, after the interpolated sample at t0); last[m] = that sample
        D_R, D_v, D_p, Dt = np.eye(3), np.zeros(3), np.zeros(3), 0.0
        snaps = [(D_R, D_v, D_p, Dt)]
        tw, ww, aw = t0, w0, a0
        for k in range(lo, hi):
            tk, wk, ak = float(t_imu[k]), w_m[k], a_m[k]
            _a_ina, a_hat, _w_inw, w_hat = correct_measurements(
                pack, b_g, b_a, ww, aw, wk, ak, True)
            D_R, D_v, D_p, Dt = state_step(D_R, D_v, D_p, Dt, tk - tw, w_hat, a_hat)
            snaps.append((D_R.copy(), D_v.copy(), D_p.copy(), Dt))
            tw, ww, aw = tk, wk, ak
        for i in idx:
            t = float(t_states[i])
            m = int(np.searchsorted(t_imu, t, side='left')) - lo   # raw samples before t
            D_R, D_v, D_p, Dt = snaps[m]
            if m == 0:
                tw, ww, aw = t0, w0, a0
            else:
                k = lo + m - 1
                tw, ww, aw = float(t_imu[k]), w_m[k], a_m[k]
            w1 = _interp_vec([t], t_imu, w_m)[0]
            a1 = _interp_vec([t], t_imu, a_m)[0]
            _a_ina, a_hat, _w_inw, w_hat = correct_measurements(
                pack, b_g, b_a, ww, aw, w1, a1, True)
            _D_R, D_v, _D_p, Dt = state_step(D_R, D_v, D_p, Dt, t - tw, w_hat, a_hat)
            v[i] = v0 + g_inW * Dt + R0 @ D_v
    return v


# ---------------------------------------------------------------------------
def _assign_brackets(t_frames, td_seed, t_states, max_interp_dt):
    """Bracketing state pair per frame: capture time t + td falls in
    [t_states[j-1], t_states[j]] (clipped at the ends — boundary frames get
    a slightly-extrapolated lambda). Frames whose bracket interval exceeds
    `max_interp_dt` (long base gap) or whose capture is outside coverage are
    dropped. Returns (j, dt_frame, keep)."""
    t_frames = np.asarray(t_frames, float)
    cap = t_frames + td_seed
    j = np.clip(np.searchsorted(t_states, cap), 1, len(t_states) - 1)
    dt_int = t_states[j] - t_states[j - 1]
    keep = (dt_int <= max_interp_dt) \
        & (cap > t_states[0] - 0.05) & (cap < t_states[-1] + 0.05)
    return j, t_frames - t_states[j - 1], keep


def _subdivide_states(t_states, T_states, knot_hz, max_gap):
    """Dense pose knots (cfg.knot_hz, the chord-error fix): split every state interval of
    length <= `max_gap` into n = ceil(dt * knot_hz) equal parts, seeding each
    inserted pose on the constant-twist interpolant of its neighbours (the same
    SO(3) x R^3 model the vision factor uses). Original states are kept, in
    order; longer gaps are left as they are (their frames are dropped by
    `_assign_brackets` anyway). knot_hz <= 0 returns the input unchanged.
    Returns (t_states, T_states)."""
    t_states = np.asarray(t_states, float)
    if knot_hz <= 0 or len(t_states) < 2:
        return t_states, list(T_states)
    t_out, T_out = [t_states[0]], [T_states[0]]
    for k in range(len(t_states) - 1):
        dt = t_states[k + 1] - t_states[k]
        n = int(np.ceil(dt * knot_hz - 1e-9)) if dt <= max_gap else 1
        for m in range(1, max(n, 1)):
            lam = m / n
            t_out.append(t_states[k] + lam * dt)
            T_out.append(interp_pose(T_states[k], T_states[k + 1], lam)[0])
        t_out.append(t_states[k + 1])
        T_out.append(T_states[k + 1])
    return np.asarray(t_out, float), T_out


def _upsampled_imus(cfg, t_imu, w_m, a_m, aux_imus):
    """Every IMU at solve.imu_min_rate_hz or faster, before init and every factor see it."""
    hz = float(getattr(cfg, "imu_min_rate_hz", 0.0) or 0.0)
    if hz <= 0.0:
        return t_imu, w_m, a_m, aux_imus
    method = getattr(cfg, "imu_upsample_method", "linear")
    t_imu, w_m, a_m = upsample_imu(t_imu, w_m, a_m, hz, method, "base")
    out = []
    for aux in aux_imus or []:
        t, w, a = upsample_imu(aux['t_imu'], aux['w_m'], aux['a_m'], hz, method, aux.get('name', 'aux'))
        out.append({**aux, 't_imu': t, 'w_m': w, 'a_m': a})
    return t_imu, w_m, a_m, (out if aux_imus else aux_imus)


def est_t_ga_run(cfg) -> bool:
    return bool(getattr(cfg, "estimate_t_ga", False))


def _tga_streams(cfg, t_states, t_imu, w_m, a_m, imu_factors, aux_data, aux_factors, cam_factors=()):
    """One TgaStream per IMU whose factors carry a t_ga key: the base, and each full aux."""
    sigmas = (cfg.sigma_g, cfg.sigma_bg, cfg.sigma_a, cfg.sigma_ba)
    model = _imu_model(cfg)
    base = TgaStream(O0, t_imu, w_m, a_m, sigmas, model, name="base")
    for k, fac in enumerate(imu_factors):
        base.add(fac, t_states[k], t_states[k + 1])
    # The vision-IMU factors share one segment of the base IMU. A factor without the t_ga key keeps
    # the raw accel, so its segment must not move.
    base.segments = list({id(f.seg): f.seg for _c, _j, f in cam_factors
                          if getattr(f, "tga", False)}.values())
    streams = [base]
    for s, (aux, facs) in enumerate(zip(aux_data or [], aux_factors or [])):
        if aux.get('gyro_only', False):
            continue
        st = TgaStream(_Os(s), aux['t_imu'], aux['w_m'], aux['a_m'], sigmas, model,
                       name=aux.get('name') or f"aux{s}")
        for ka, kb, fac in facs:
            st.add(fac, t_states[ka] - fac.window_td, t_states[kb] - fac.window_td)
        streams.append(st)
    return streams


def _aux_knot_pairs(t_states, aux_imu_factor_hz, aux_imu_stride=0):
    """Decimated knot set + consecutive (ka, kb) intervals for aux-IMU factors.

    Resolution order:
      1. aux_imu_stride > 0  → use directly (wins over rate-based).
      2. aux_imu_factor_hz > 0 → stride = round((1/rate)/median_state_dt).
      3. else → stride 1 (one factor per base interval).

    Note: the helper param default is 0 (no decimation) so that direct unit
    tests calling _aux_knot_pairs(t, 0.0) still get stride-1. The config field
    CamImuCalibConfig.aux_imu_stride defaults to 2 (different on purpose).

    The first (0) and last (K) knots are always included; a non-divisible tail
    yields one shorter final interval."""
    n = len(t_states)
    K = n - 1
    if aux_imu_stride and aux_imu_stride > 0:
        stride = int(aux_imu_stride)
    elif aux_imu_factor_hz and aux_imu_factor_hz > 0:
        dt = float(np.median(np.diff(np.asarray(t_states, float))))
        stride = max(1, int(round((1.0 / aux_imu_factor_hz) / dt)))
    else:
        stride = 1
    stride = max(1, stride)
    knots = list(range(0, n, stride))
    if knots[-1] != K:
        knots.append(K)
    return knots, list(zip(knots[:-1], knots[1:]))


def build_joint_graph(t_states, frames_per_cam, T_init, v_init,
                      seed: CamImuSeed, calib_bases, image_sizes,
                      t_imu, w_m, a_m, cfg: CamImuCalibConfig,
                      aux_data=None, rolling_modes=None, pin_trs=None,
                      pin_intr=None, pin_extrinsics=None,
                      structure="board", landmarks_init=None,
                      board_ids_per_cam=None, board_poses_init=None,
                      frame_dt_cams=None, board_pids_per_cam=None,
                      board_points=None, vision_pose_cams=None):
    """Assemble graph + initial Values.

    `structure="board"` (default): `frames_per_cam[c]` lists (uv, pts, j,
    dt_frame); each frame is constrained by the bracketing states [X(j-1), X(j)]
    through CameraCalibInterpFactor over the KNOWN points `pts` + the shared
    board pose `B0`.

    `structure="feature"`: `frames_per_cam[c]` lists (uv, ids, j, dt_frame);
    each corner reprojects a WORLD-FRAME `Point3` landmark variable `L(id)`
    (CameraFeatureInterpFactor). `landmarks_init` is the dict {id: (3,) world}
    used to seed the landmark variables (+ a weak prior for regularization);
    there is no board pose.

    Board shape (`cfg.estimate_board_heights`, board mode, single board): the
    corners' board point ids per frame (`board_pids_per_cam`, parallel to
    `frames_per_cam`) and the full board point array (`board_points`) let both
    vision factors take the height key H0 (+ prior / gauge).

    `vision_pose_cams`: per-camera vision factor ("interp" | "imu"); None = every
    camera uses `cfg.vision_pose`. Lets a rig mix them (e.g. the IMU-propagated
    factor for rolling-shutter cameras, interpolation for global-shutter ones).

    Returns (graph, values, cam_factors, imu_factors, aux_factors);
    cam_factors entries are (c, j, factor)."""
    g_inW = np.array([0.0, 0.0, -cfg.gravity])
    K = len(t_states) - 1
    n_cams_g = len(frames_per_cam)
    feature = (structure == "feature")
    if rolling_modes is None:
        rolling_modes = [cfg.rolling_mode] * n_cams_g
    if pin_trs is None:
        pin_trs = [cfg.pin_tr] * n_cams_g
    if pin_intr is None:
        pin_intr = [False] * n_cams_g
    # cam-IMU extrinsic pin (unified-estimate U3): default from cfg.estimate_extrinsics
    if pin_extrinsics is None:
        pin_extrinsics = [not getattr(cfg, "estimate_extrinsics", True)] * n_cams_g
    pin_td_base = not getattr(cfg, "estimate_td", True)   # base camera <-> IMU td
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    if feature:
        if not landmarks_init:
            raise ValueError("structure='feature' needs landmarks_init")
        for lid, p in landmarks_init.items():
            values.insert(_L(int(lid)), gtsam.Point3(*np.asarray(p, float)))
    elif board_poses_init:
        for b, T in board_poses_init.items():
            values.insert(_B(int(b)), T)
    else:
        values.insert(B0, seed.T_BinW)
    est_heights = bool(getattr(cfg, "estimate_board_heights", False))
    n_bp = 0
    if est_heights:
        if feature or board_poses_init or board_ids_per_cam is not None:
            raise ValueError("estimate board heights: board mode with ONE board only")
        if board_pids_per_cam is None or board_points is None:
            raise ValueError("estimate board heights needs board_pids_per_cam + board_points")
        n_bp = len(board_points)
        values.insert(H0, np.zeros(n_bp))
        for f in _board_height_factors(board_points,
                                       float(getattr(cfg, "board_height_sigma", 0.003))):
            graph.add(f)
    values.insert(I0, np.zeros(24))
    est_t_ga = bool(getattr(cfg, "estimate_t_ga", False))
    if est_t_ga:
        values.insert(O0, np.zeros(1))
    for c in range(len(frames_per_cam)):
        values.insert(_Cc(c), np.zeros(calib_bases[c].DIM))

    for k in range(K + 1):
        values.insert(X(k), T_init[k])
        values.insert(_Vk(k), np.concatenate([v_init[k], seed.b_g, seed.b_a]))

    cam_factors = []
    if vision_pose_cams is None:
        vision_pose_cams = [getattr(cfg, "vision_pose", "imu")] * len(frames_per_cam)
    for vp in vision_pose_cams:
        if vp not in ("interp", "imu"):
            raise ValueError(f"vision_pose must be 'interp' or 'imu', got {vp!r}")
    vision_imu_cams = [(vp == "imu") and not feature for vp in vision_pose_cams]
    vision_imu = any(vision_imu_cams)
    if vision_imu:
        imu_segment = ImuSegment(t_imu, w_m, a_m, _imu_model(cfg))
        imu_gyro = getattr(cfg, "vision_imu_gyro", "calibrated")
        imu_dw_column = bool(getattr(cfg, "vision_imu_dw_column", True))
        imu_nearest = getattr(cfg, "vision_imu_anchor", "lower") == "nearest"
        imu_tga = (est_t_ga and imu_gyro == "calibrated"
                   and bool(getattr(cfg, "vision_imu_t_ga", True)))
        frame_dt_sigma = float(getattr(cfg, "vision_imu_frame_dt_sigma", 0.0) or 0.0)
        if frame_dt_sigma > 0:
            # Per-frame offset mode (diagnostics, sensor-clock pass 1) leaves the t_ga
            # term out: O0 in every frame factor beside the delta_f keys changes the
            # elimination order, and Cholesky then fails on the _TINY priors (LM stops
            # at iteration 0). That pass only needs the offsets; pass 2 has the term.
            imu_tga = False
        frame_dt_noise = (gtsam.noiseModel.Isotropic.Sigma(1, frame_dt_sigma)
                          if frame_dt_sigma > 0 else None)
        n_imu_uncovered = 0
        n_frame_dt = 0
    for c, frames in enumerate(frames_per_cam):
        # per-frame board id (board mode w/ multiple boards); default = anchor B0
        bids = (board_ids_per_cam[c] if board_ids_per_cam is not None
                else [0] * len(frames))
        # corner board-point ids per frame: the heights factors use them, the report's corner
        # heatmap reads them back from `fac.report_pids` either way
        pids_c = (board_pids_per_cam[c] if board_pids_per_cam is not None
                  else [None] * len(frames))
        hk = dict(board_pids=None, n_board_points=0)
        for (uv, pts, j, dt_frame), bid, pids in zip(frames, bids, pids_c):
            if est_heights:
                hk = dict(board_pids=pids, n_board_points=n_bp)
            if feature:
                ids = [int(i) for i in pts]
                fac = CameraFeatureInterpFactor(
                    uv, ids, dt_frame, t_states[j] - t_states[j - 1],
                    calib_bases[c], image_sizes[c],
                    rolling_mode=rolling_modes[c])
                graph.add(fac.make_factor(
                    X(j - 1), X(j), _Cc(c), [_L(i) for i in ids],
                    sigma_px=cfg.sigma_px, huber_px=cfg.huber_px))
            elif vision_imu_cams[c]:
                # IMU-propagated pose (E8): anchor on X(j-1) (or the nearer
                # bracketing state) and integrate the raw IMU to each corner's
                # capture time. Frames whose IMU window leaves the record are
                # dropped here (the interpolation would clamp silently).
                frame_dt_on = frame_dt_noise is not None and (
                    frame_dt_cams is None or bool(frame_dt_cams[c]))
                dt_int = t_states[j] - t_states[j - 1]
                cap = dt_frame + calib_bases[c].td      # seed capture offset from X(j-1)
                k = j if (imu_nearest and cap > 0.5 * dt_int) else j - 1
                t_cap = t_states[j - 1] + cap
                if not (imu_segment.covers(t_states[k], t_states[k])
                        and imu_segment.covers(t_cap - 0.05, t_cap + 0.05)):
                    n_imu_uncovered += 1
                    continue
                fac = CameraCalibImuFactor(
                    uv, pts, dt_frame - (t_states[k] - t_states[j - 1]),
                    imu_segment, t_states[k], g_inW, calib_bases[c],
                    image_sizes[c], rolling_mode=rolling_modes[c],
                    gyro=imu_gyro, dw_column=imu_dw_column,
                    frame_dt=frame_dt_on, tga=imu_tga,
                    linearized=bool(getattr(cfg, "vision_imu_linearized", False)), **hk)
                k_dt = None
                if frame_dt_on:
                    k_dt = _Fdt(n_frame_dt)
                    n_frame_dt += 1
                    values.insert(k_dt, np.zeros(1))
                    graph.add(gtsam.PriorFactorVector(k_dt, np.zeros(1), frame_dt_noise))
                graph.add(fac.make_factor(
                    X(k), _Vk(k), _B(int(bid)), _Cc(c),
                    k_imu=(I0 if fac.uses_imu_key else None), k_dt=k_dt,
                    k_heights=(H0 if est_heights else None),
                    sigma_px=cfg.sigma_px, huber_px=cfg.huber_px,
                    k_tga=(O0 if imu_tga else None)))
            else:
                fac = CameraCalibInterpFactor(
                    uv, pts, dt_frame, t_states[j] - t_states[j - 1],
                    calib_bases[c], image_sizes[c],
                    rolling_mode=rolling_modes[c], **hk)
                graph.add(fac.make_factor(X(j - 1), X(j), _B(int(bid)), _Cc(c),
                                          k_heights=(H0 if est_heights else None),
                                          sigma_px=cfg.sigma_px,
                                          huber_px=cfg.huber_px))
            if not feature:
                fac.report_pids = pids
            cam_factors.append((c, j, fac))
    if vision_imu and n_imu_uncovered:
        logger.info("vision_pose=imu: dropped %d frames outside IMU coverage",
                  n_imu_uncovered)

    # Adaptive per-camera depth-margin guard (feature mode): set each factor's
    # z_min from the seed scene scale — z_min = frac * median(seed depth), floored
    # — so the grazing guard scales with the scene instead of a hard-coded metre.
    if feature and getattr(cfg, "z_min_frac", 0.0) > 0.0:
        per_cam = {c: [] for c in range(n_cams_g)}
        for c, j, fac in cam_factors:
            P = np.array([landmarks_init[i] for i in fac.ids])
            per_cam[c].append(fac.seed_depths(T_init[j - 1], T_init[j], P))
        for c in range(n_cams_g):
            if not per_cam[c]:
                continue
            d = np.concatenate(per_cam[c])
            d = d[d > 0]                         # ignore behind-camera (cheirality)
            if d.size == 0:
                continue
            z_min = max(cfg.z_min_frac * float(np.median(d)), cfg.z_min_floor)
            for cc, _j, fac in cam_factors:
                if cc == c:
                    fac.z_min = z_min

    imu_factors = []
    for k in range(K):
        pre = ACIPreintegrator(cfg.sigma_g, cfg.sigma_bg, cfg.sigma_a,
                               cfg.sigma_ba, model=_imu_model(cfg), avg_meas=True)
        pre.reset(seed.b_g, seed.b_a)
        pre.set_intrinsics(create_imu_model(_imu_model(cfg)))
        t, w, a = _slice_imu(t_imu, w_m, a_m, t_states[k], t_states[k + 1])
        pre.feed_imu_interval(t, w, a)
        fac = ImuFactorACI_In.from_preintegrator(pre, g_inW)
        graph.add(fac.make_factor(X(k), X(k + 1), _Vk(k), _Vk(k + 1), I0,
                                  k_tga=(O0 if est_t_ga else None)))
        imu_factors.append(fac)

    # --- auxiliary IMUs: own [bg, ba, v] states + Multi_In_V factors.
    # Aux pose at endpoint k = base pose at t_k propagated by td, so the aux
    # preintegration window IN AUX STAMPS is the same [t_k, t_{k+1}].
    aux_factors = []
    for s, aux in enumerate(aux_data or []):
        sd = aux['seed']
        facs = []
        if aux.get('gyro_only', False):
            # GYRO-ONLY aux (Multi_In_SO3): the aux is calibrated from its
            # GYRO alone -> extrinsic ROTATION + td + Dw, no lever arm / Da.
            # Avoids the accel entirely: no gyro-accel timing offset (the
            # t265 BMI055 interpolation issue), no noisy cheap-accel. State
            # is the aux gyro bias bg(3) per endpoint; calib ξ31 with only
            # rotation/td/Dw free, translation+Da+Tg+R pinned (gyro can't
            # observe them).
            kk, pairs = _aux_knot_pairs(t_states, getattr(cfg, "aux_imu_factor_hz", 0.0),
                                        getattr(cfg, "aux_imu_stride", 0))
            for k in kk:
                values.insert(_BGs(s, k), np.asarray(sd.b_g_s, float))
            values.insert(_As(s), np.zeros(31))
            for ka, kb in pairs:
                pre = ACIGyroPreintegrator(cfg.sigma_g, cfg.sigma_bg,
                                           model=_imu_model(cfg), avg_meas=True)
                pre.reset(sd.b_g_s)
                pre.set_intrinsics(create_imu_model(_imu_model(cfg)))
                t, w, _a = _slice_imu(aux['t_imu'], aux['w_m'], aux['a_m'],
                                      t_states[ka] - sd.td, t_states[kb] - sd.td)
                pre.feed_imu_interval(t, w)
                fac = ImuFactorACI_Multi_In_SO3.from_preintegrator(
                    pre, sd.T_IstoI, sd.td, window_td=sd.td)
                graph.add(fac.make_factor(X(ka), X(kb),
                                          _BGs(s, ka), _BGs(s, kb), _As(s)))
                facs.append((ka, kb, fac))
            graph.add(gtsam.PriorFactorVector(
                _BGs(s, 0), np.asarray(sd.b_g_s, float),
                gtsam.noiseModel.Isotropic.Sigma(3, cfg.sigma_bg0)))
            # calib ξ31: free rotation(0:3) + td(6) + gyro Dw; pin the rest.
            # Gyro-only aux observes Dw alone, so free just the 'dw' block
            # (n_dw cols at offset 7) regardless of stage.
            sig_A = np.full(31, _TINY)
            if getattr(cfg, "estimate_aux_extrinsics", True):
                sig_A[0:3] = _BIG          # extrinsic rotation (gyro sees rot only)
            if getattr(cfg, "estimate_aux_td", True):
                sig_A[6] = _BIG            # inter-IMU td
            if cfg.imu_intrinsics != "pinned":
                sig_A[7:31] = intrinsic_sigma_vector(
                    _imu_model(cfg), cfg.imu_intrinsics, sigma_dw_da=cfg.sigma_dw_da,
                    sigma_tg=cfg.sigma_tg, sigma_R=cfg.sigma_imu_R, tiny=_TINY,
                    freed=["dw"])
            graph.add(gtsam.PriorFactorVector(
                _As(s), np.zeros(31), gtsam.noiseModel.Diagonal.Sigmas(sig_A)))
            aux_factors.append(facs)
            continue

        # --- full aux (Multi_In_V): own [bg, ba, v] states + lever arm ---
        kk, pairs = _aux_knot_pairs(t_states, getattr(cfg, "aux_imu_factor_hz", 0.0),
                                    getattr(cfg, "aux_imu_stride", 0))
        for k in kk:
            values.insert(_VBs(s, k), np.concatenate(
                [sd.b_g_s, np.zeros(3), v_init[k]]))      # [bg, ba, v]
        values.insert(_As(s), np.zeros(31))
        est_tga_s = est_t_ga and not aux.get('gyro_only', False)
        if est_tga_s:                    # aux IMU's OWN gyro<->accel offset
            values.insert(_Os(s), np.zeros(1))
        for ka, kb in pairs:
            pre = ACIPreintegrator(cfg.sigma_g, cfg.sigma_bg, cfg.sigma_a,
                                   cfg.sigma_ba, model=_imu_model(cfg), avg_meas=True)
            pre.reset(sd.b_g_s, np.zeros(3))
            pre.set_intrinsics(create_imu_model(_imu_model(cfg)))
            # window-shift by the init td_s: slice the aux at base stamps
            # MINUS sd.td so the aux preint physically aligns with the base
            # interval; the factor then bridges only the residual (robust to
            # LARGE absolute aux-base offsets — keeps Exp(w*dt) propagation tiny)
            t, w, a = _slice_imu(aux['t_imu'], aux['w_m'], aux['a_m'],
                                 t_states[ka] - sd.td, t_states[kb] - sd.td)
            pre.feed_imu_interval(t, w, a)
            fac = ImuFactorACI_Multi_In_V.from_preintegrator(
                pre, g_inW, sd.T_IstoI, sd.td, window_td=sd.td)
            graph.add(fac.make_factor(X(ka), X(kb), _Vk(ka), _Vk(kb),
                                      _VBs(s, ka), _VBs(s, kb), _As(s),
                                      k_tga=(_Os(s) if est_tga_s else None)))
            facs.append((ka, kb, fac))     # keep the (decimated) knot pair
        aux_factors.append(facs)
        # priors: vbs(0) [bg, ba, v] + ImuCalib staging (extrinsic/td free;
        # aux intrinsics staged like the base)
        sig_vbs0 = np.concatenate([np.full(3, cfg.sigma_bg0),
                                   np.full(3, cfg.sigma_ba0),
                                   np.full(3, cfg.sigma_v0)])
        graph.add(gtsam.PriorFactorVector(
            _VBs(s, 0), np.concatenate([sd.b_g_s, np.zeros(3), v_init[0]]),
            gtsam.noiseModel.Diagonal.Sigmas(sig_vbs0)))
        sig_A = np.full(31, _BIG)               # extrinsic + td free (default)
        if not getattr(cfg, "estimate_aux_extrinsics", True):
            sig_A[0:6] = _TINY                  # hold aux-IMU extrinsic T_IstoI
        if not getattr(cfg, "estimate_aux_td", True):
            sig_A[6] = _TINY                    # hold inter-IMU td at seed
        sig_A[7:31] = intrinsic_sigma_vector(
            _imu_model(cfg), cfg.imu_intrinsics, sigma_dw_da=cfg.sigma_dw_da,
            sigma_tg=cfg.sigma_tg, sigma_R=cfg.sigma_imu_R, tiny=_TINY)
        graph.add(gtsam.PriorFactorVector(
            _As(s), np.zeros(31), gtsam.noiseModel.Diagonal.Sigmas(sig_A)))
        if est_tga_s:                    # weak prior, same scale as the base
            graph.add(gtsam.PriorFactorVector(
                _Os(s), np.zeros(1),
                gtsam.noiseModel.Isotropic.Sigma(1, float(cfg.sigma_t_ga))))

    # --- gauge + priors ---
    graph.add(GaugeFactorPose3(T_init[0]).make_factor(X(0)))
    # feature mode: weak isotropic prior anchoring each landmark at its metric
    # seed — regularizes weakly-observed landmarks (few views / low parallax)
    # without biasing well-constrained ones. Scale is observable from the IMU,
    # so this is a Tikhonov stabilizer, not a scale gauge.
    if feature:
        sig_L = float(getattr(cfg, "sigma_landmark0", 1.0))
        noise_L = gtsam.noiseModel.Isotropic.Sigma(3, sig_L)
        for lid, p in landmarks_init.items():
            graph.add(gtsam.PriorFactorPoint3(
                _L(int(lid)), gtsam.Point3(*np.asarray(p, float)), noise_L))
    vb0 = np.concatenate([v_init[0], seed.b_g, seed.b_a])
    sig_vb0 = np.concatenate([np.full(3, cfg.sigma_v0),
                              np.full(3, cfg.sigma_bg0),
                              np.full(3, cfg.sigma_ba0)])
    graph.add(gtsam.PriorFactorVector(
        _Vk(0), vb0, gtsam.noiseModel.Diagonal.Sigmas(sig_vb0)))
    # IMU intrinsics staging: the 24-dim tangent ξ_in around the IDENTITY base
    # used in the preintegrations (layout per the model — see imu_intrinsics).
    # Freed blocks get a weak prior (regularizes near-null directions on weak
    # excitation); unknown stages raise inside intrinsic_sigma_vector.
    sig_I = intrinsic_sigma_vector(
        _imu_model(cfg), cfg.imu_intrinsics, sigma_dw_da=cfg.sigma_dw_da,
        sigma_tg=cfg.sigma_tg, sigma_R=cfg.sigma_imu_R, tiny=_TINY)
    graph.add(gtsam.PriorFactorVector(
        I0, np.zeros(24), gtsam.noiseModel.Diagonal.Sigmas(sig_I)))
    if est_t_ga:
        graph.add(gtsam.PriorFactorVector(
            O0, np.zeros(1),
            gtsam.noiseModel.Isotropic.Sigma(1, float(cfg.sigma_t_ga))))
    # camera bundles: free, except tr pinned on global-shutter cameras,
    # td anchored at the seed (time-gauge common mode; see config docstring),
    # and intrinsics pinned for cameras with pin_intrinsics (diagnostic /
    # known-intrinsics workflows — holds ξ8:dim at the seed value)
    for c in range(n_cams_g):
        if not (pin_trs[c] or cfg.sigma_td_prior > 0 or pin_intr[c]
                or pin_extrinsics[c] or pin_td_base):
            continue
        dim = calib_bases[c].DIM
        sig_C = np.full(dim, _BIG)
        if pin_extrinsics[c]:
            sig_C[0:6] = _TINY            # hold the cam-IMU extrinsic T_C->I at seed
        if pin_td_base:
            sig_C[6] = _TINY              # hard-pin the camera <-> IMU td at seed
        elif cfg.sigma_td_prior > 0:
            sig_C[6] = cfg.sigma_td_prior  # soft anchor (mutually exclusive w/ pin)
        if pin_trs[c]:
            sig_C[7] = _TINY
        if pin_intr[c]:
            sig_C[8:dim] = _TINY          # all intrinsics (8 .. 8+n_intr)
        graph.add(gtsam.PriorFactorVector(
            _Cc(c), np.zeros(dim),
            gtsam.noiseModel.Diagonal.Sigmas(sig_C)))
    if getattr(cfg, "bound_intrinsics", False):
        for c in range(n_cams_g):
            if not pin_intr[c]:
                fac = intr_bound_factor(_Cc(c), calib_bases[c])
                if fac is not None:
                    graph.add(fac)

    return graph, values, cam_factors, imu_factors, aux_factors


# ---------------------------------------------------------------------------
def _vision_imu_diagnostics(cam_factors, sols, t_states, marg, imu_model, res=None):
    """Diagnostics for vision_pose="imu" (E8): the propagation windows Δ at
    the solved td/tr (ms; sign, length relative to the state interval) and the
    marginal correlations between the gyro-scale diagonal and each camera's
    td / tr — the collinearity the Dw column could introduce."""
    from mvis.imu import dw_matrix, n_dw_for
    out = {}
    dl, beyond = [], 0
    for c, j, fac in cam_factors:
        if not isinstance(fac, CameraCalibImuFactor):
            continue
        d = fac.deltas(sols[c])
        dl.append(d)
        beyond += int(np.sum(np.abs(d) > (t_states[j] - t_states[j - 1])))
    if res is not None:
        sel = [(c, f) for c, _j, f in cam_factors
               if isinstance(f, CameraCalibImuFactor) and f.frame_dt]
        fdt = [float(res.atVector(f.key_dt)[0]) * 1e3 for _c, f in sel]
        if fdt:
            fdt = np.asarray(fdt)
            # per-frame series (stamp time, camera, offset ms) for timing-pattern analysis
            out['frame_dt_series'] = [[float(f.t_anchor + f.dt_anchor), int(c), float(x)]
                                      for (c, f), x in zip(sel, fdt)]
            out['frame_dt_ms'] = {
                'n': len(fdt), 'mean': float(fdt.mean()), 'std': float(fdt.std()),
                'mad_sigma': float(1.4826 * np.median(np.abs(fdt - np.median(fdt)))),
                'p5': float(np.percentile(fdt, 5)), 'p95': float(np.percentile(fdt, 95))}
    if dl:
        d = np.concatenate(dl) * 1e3
        out['delta_ms'] = {'min': float(d.min()), 'median': float(np.median(d)),
                           'max': float(d.max()), 'abs_mean': float(np.abs(d).mean()),
                           'frac_negative': float(np.mean(d < 0)),
                           'frac_beyond_interval': beyond / len(d)}
    if marg is None:
        return out
    # tangent indices of x_dw that move the Dw DIAGONAL (model-agnostic)
    n = n_dw_for(imu_model)
    ident = dw_matrix(np.zeros(n), imu_model)
    diag_idx = []
    for i in range(n):
        e = np.zeros(n); e[i] = 1e-3
        dd = np.diag(dw_matrix(e, imu_model) - ident)
        if np.abs(dd).max() > 0:
            diag_idx.append((int(np.argmax(np.abs(dd))), i))
    try:
        corr = {}
        for c in range(len(sols)):
            kv = gtsam.KeyVector([I0, _Cc(c)])
            P = marg.jointMarginalCovariance(kv).fullMatrix()
            s = np.sqrt(np.diag(P))
            C = P / np.outer(s, s)
            corr[f'cam{c}'] = {
                'dw_td': [float(C[i, 24 + 6]) for _ax, i in sorted(diag_idx)],
                'dw_tr': [float(C[i, 24 + 7]) for _ax, i in sorted(diag_idx)]}
        out['dw_corr'] = corr
    except RuntimeError:
        pass
    return out


def _legacy_frame_args(c, j, values):
    """`error` arguments of a legacy (`keys is None`) board-mode vision factor at `values`:
    [X(j-1), X(j), B0, C(c)]. Factors built through make_factor know their own keys (interp AND
    IMU-propagated, any board) and go through their `*_values` methods instead."""
    return (values.atPose3(X(j - 1)), values.atPose3(X(j)),
            values.atPose3(B0), values.atVector(_Cc(c)))


def _frame_corner_errors(fac, c, j, values):
    """Per-corner signed (u, v) errors (n, 2) of one board-mode vision factor at `values`, NaN rows
    for guard-skipped corners; `_frame_corner_norms` is their row norms."""
    if getattr(fac, "keys", None) is not None:
        return fac.corner_errors_values(values)
    return fac.corner_errors(*_legacy_frame_args(c, j, values))


def _frame_corner_norms(fac, c, j, values):
    """Per-corner error norms of one board-mode vision factor at `values`."""
    if getattr(fac, "keys", None) is not None:
        return fac.corner_norms_values(values)
    return fac.corner_norms(*_legacy_frame_args(c, j, values))


class CornerTable(NamedTuple):
    """Per-corner reprojection errors of every board-mode vision factor at ONE Values
    (a speed-up): one projection pass feeds the per-frame RMS series, the pooled
    Kalibr-convention population and the report's per-camera corner data
    (`by_camera`), instead of one pass each.

    `e` (n_corner,): corner norms in px, NaN for guard-skipped corners;
    `offsets` (n_factors + 1,): CSR offsets of each factor's corners in `e`;
    `frame_cam` (n_factors,): camera index per factor;
    `corner_cam` (n_corner,): camera index per corner;
    `vec` (n_corner, 2): the signed (u, v) errors whose row norms are `e`, NaN rows for
    guard-skipped corners; None when some factor has no `corner_errors` (norms only)."""
    e: np.ndarray
    offsets: np.ndarray
    frame_cam: np.ndarray
    corner_cam: np.ndarray
    vec: np.ndarray | None = None

    def by_camera(self, cam_factors) -> list[dict]:
        """Per camera, the report's corner data: `vectors_by_frame` {frame index within the
        camera: (n, 2) finite signed errors} and `corner_errors` {board point id: [norms]}
        (ids from each factor's `report_pids`; empty when the factor has none). With ids,
        also `n_tags`: distinct (frame j, AprilGrid tag `pid // 4`) pairs over the finite
        corners, keyed by frame rather than factor so a frame split into sub-frame factors
        counts each tag once (only meaningful on an AprilGrid)."""
        out = [dict(vectors_by_frame={}, corner_errors={}) for _ in range(int(self.frame_cam.max()) + 1)] \
            if len(self.frame_cam) else []
        if self.vec is None:
            return out
        n_seen = [0] * len(out)
        tags = [set() for _ in out]
        for k, (c, j, fac) in enumerate(cam_factors):
            sl = slice(self.offsets[k], self.offsets[k + 1])
            v, e = self.vec[sl], self.e[sl]
            ok = np.isfinite(e)
            if ok.any():
                out[c]["vectors_by_frame"][n_seen[c]] = v[ok]
            n_seen[c] += 1
            pids = getattr(fac, "report_pids", None)
            if pids is not None and len(pids) == len(e):
                ce = out[c]["corner_errors"]
                for pid, x in zip(np.asarray(pids, int)[ok], e[ok]):
                    ce.setdefault(int(pid), []).append(float(x))
                    tags[c].add((int(j), int(pid) // 4))
                out[c]["n_tags"] = len(tags[c])
        return out

    def frame_rms(self):
        """Per-frame RMS of the corner norms (px) plus the camera index per
        frame — `_reproj_rms`'s output. Guard-skipped corners are excluded; a
        frame with none valid is NaN. Same expression as the two-pass code on
        each factor's (contiguous) slice, so the result is bitwise identical."""
        out = np.empty(len(self.frame_cam))
        for k in range(len(self.frame_cam)):
            e = self.e[self.offsets[k]:self.offsets[k + 1]]
            out[k] = (np.sqrt(np.nanmean(e ** 2))
                      if np.isfinite(e).any() else np.nan)
        return out, self.frame_cam.copy()

    def pooled(self):
        """Finite corner norms of ALL frames concatenated in factor order plus
        the camera index per corner — `_reproj_corner_norms`'s output."""
        v = np.isfinite(self.e)
        return self.e[v], self.corner_cam[v]


def _corner_norm_table(cam_factors, values) -> CornerTable:
    """One corner pass over `cam_factors` ((c, j, factor) entries) at `values`; see
    `CornerTable`. When every factor has `corner_errors`, the signed errors come from
    `_frame_corner_errors` and the norms are their row norms (bitwise `corner_norms`; `vec` is
    (0, 2) for no factors). Otherwise (the tests/_reference doubles) it falls back to
    `_frame_corner_norms` and `vec` is None. Both handle the legacy (`keys is None`) factors."""
    signed = all(hasattr(fac, "corner_errors") for _c, _j, fac in cam_factors)
    if signed:
        vecs = [_frame_corner_errors(fac, c, j, values) for c, j, fac in cam_factors]
        norms = [np.linalg.norm(v, axis=1) for v in vecs]
    else:
        vecs, norms = [], [_frame_corner_norms(fac, c, j, values) for c, j, fac in cam_factors]
    frame_cam = np.array([c for c, _j, _fac in cam_factors], int)
    counts = np.array([len(e) for e in norms], int)
    offsets = np.zeros(len(cam_factors) + 1, int)
    np.cumsum(counts, out=offsets[1:])
    e = np.concatenate(norms) if norms else np.empty(0)
    vec = np.concatenate(vecs) if vecs else (np.empty((0, 2)) if signed else None)
    return CornerTable(np.asarray(e, float), offsets, frame_cam,
                       np.repeat(frame_cam, counts), vec)


def _reproj_rms(cam_factors, values, table=None):
    """Per-frame RMS of per-corner error norms (px) at the given Values, plus
    the camera index per frame (cam_factors entries are (c, j, factor)).
    Guard-skipped corners are excluded; a frame with none valid is NaN.
    `table`: a `_corner_norm_table` already built at `values` (no new pass)."""
    if table is None:
        table = _corner_norm_table(cam_factors, values)
    return table.frame_rms()


def _reproj_corner_norms(cam_factors, values, table=None):
    """Kalibr-convention population for reprojection stats: per-corner error
    norms (px) of ALL frames concatenated, plus the camera index per corner.
    Guard-skipped (NaN) corners are dropped from the pool.
    `table`: a `_corner_norm_table` already built at `values` (no new pass)."""
    if table is None:
        table = _corner_norm_table(cam_factors, values)
    return table.pooled()


def _imu_cost_split(imu_factors, values, k_tga=None):
    """Whitened squared cost split by residual rows [theta,p,v,bg,ba] and the
    per-interval total (the long-gap canary). `k_tga` holds the estimated t_ga, so the cost is the
    one the solver saw; None means the run has no t_ga."""
    t_ga = float(values.atVector(k_tga)[0]) if k_tga is not None else 0.0
    blocks = {'theta': slice(0, 3), 'p': slice(3, 6), 'v': slice(6, 9),
              'bg': slice(9, 12), 'ba': slice(12, 15)}
    sums = {name: 0.0 for name in blocks}
    per_interval = np.empty(len(imu_factors))
    for k, fac in enumerate(imu_factors):
        r = fac.whitened_error(values.atPose3(X(k)), values.atPose3(X(k + 1)),
                               values.atVector(_Vk(k)),
                               values.atVector(_Vk(k + 1)),
                               values.atVector(I0), t_ga)
        per_interval[k] = float(r @ r)
        for name, sl in blocks.items():
            sums[name] += float(r[sl] @ r[sl])
    return sums, per_interval


IMU_BLOCKS = (("theta", slice(0, 3)), ("p", slice(3, 6)), ("v", slice(6, 9)),
              ("bg", slice(9, 12)), ("ba", slice(12, 15)))


def _imu_block_residuals(imu_factors, values, k_tga=None) -> dict:
    """Per interval and per IMU-state block [theta, p, v, bg, ba]: the whitened norm |W r_b| and the raw
    norm |r_b| in the block's unit (rad, m, m/s and the bias steps), with each interval's Dt. The report's
    IMU Residuals section reads it; same factors and t_ga as `_imu_cost_split`. A factor without the raw
    residual (`_common`) leaves its raw row NaN."""
    t_ga = float(values.atVector(k_tga)[0]) if k_tga is not None else 0.0
    n = len(imu_factors)
    white = np.full((n, len(IMU_BLOCKS)), np.nan)
    raw = np.full((n, len(IMU_BLOCKS)), np.nan)
    dt = np.full(n, np.nan)
    for k, fac in enumerate(imu_factors):
        args = (values.atPose3(X(k)), values.atPose3(X(k + 1)), values.atVector(_Vk(k)),
                values.atVector(_Vk(k + 1)), values.atVector(I0), t_ga)
        r = fac.whitened_error(*args)
        e = fac._common(*args)[0] if hasattr(fac, "_common") else None
        for j, (_name, sl) in enumerate(IMU_BLOCKS):
            white[k, j] = float(np.linalg.norm(r[sl]))
            if e is not None:
                raw[k, j] = float(np.linalg.norm(e[sl]))
        dt[k] = float(getattr(fac, "Dt", np.nan))
    return {"blocks": [b for b, _ in IMU_BLOCKS], "white": white, "raw": raw, "dt": dt}


# ---------------------------------------------------------------------------
# old private name of detection.corner_subsample.corner_mask, kept for the tests that import it
_corner_mask = corner_mask


def _decimate(t, cap_hz):
    """Keep-mask thinning timestamps to at most cap_hz.

    Keeps a sample once it is at least 0.75/cap_hz after the last kept one. The
    0.75 jitter tolerance is essential: with a hard `>= 1/cap_hz` threshold a
    source already at ~cap_hz (e.g. a 30 Hz camera, state_hz=30) aliases to
    cap_hz/2 — every frame landing a hair early from timing jitter is dropped, so
    the next (double-spaced) one is kept. Tolerating up to 25% early keeps such a
    same-rate source intact (true 30 Hz states) while a genuine >=1.33x oversampled
    source is still thinned toward cap_hz."""
    keep = np.ones(len(t), bool)
    if cap_hz > 0:
        min_dt, last = 0.75 / cap_hz, -np.inf
        for i, ti in enumerate(t):
            keep[i] = (ti - last) >= min_dt
            if keep[i]:
                last = ti
    return keep


def _cam_factor_hz(cam, cfg):
    """Per-camera vision-FACTOR rate. Prefers the cam-dict
    `factor_hz` (legacy `target_hz` accepted), then `cfg.factor_hz` (modern default
    10 Hz — decoupled from the trajectory rate), and only falls back to `cfg.state_hz`
    if `factor_hz` is explicitly None. Same concept as `CameraConfig.factor_hz`."""
    hz = cam.get('factor_hz')
    if hz is None:
        hz = cam.get('target_hz')        # legacy cam-dict key
    if hz is None:
        hz = cfg.factor_hz
    return cfg.state_hz if hz is None else hz


def _cam_pin_tr(cam, cfg):
    """Per-camera readout-pinning for the solve. Prefers the canonical cam-dict key
    `estimate_readout` (True = rolling, estimate tr), falls back to the legacy
    `pin_tr` key, else the config default (`cfg.pin_tr` = not cfg.estimate_readout)."""
    if 'estimate_readout' in cam:
        return not bool(cam['estimate_readout'])
    if 'pin_tr' in cam:
        return bool(cam['pin_tr'])
    return cfg.pin_tr


def _cam_pin_extr(cam, cfg):
    """Per-camera cam-IMU extrinsic pinning (unified-estimate U3). Prefers the
    cam-dict key `estimate_extrinsics` (True = estimate T_C->I), else the config
    default (`cfg.estimate_extrinsics`). True return = pin (hold at seed)."""
    if 'estimate_extrinsics' in cam:
        return not bool(cam['estimate_extrinsics'])
    return not bool(getattr(cfg, "estimate_extrinsics", True))


def _cam_vision_pose(cam, cfg) -> str:
    """Vision factor of one camera: its own `vision_pose` key, else solve.vision_pose."""
    return cam.get('vision_pose') or getattr(cfg, "vision_pose", "imu")


def _marginal_sigmas(graph, res, calib_bases, n_cams, cfg):
    """(marg, per-camera calib sigmas, IMU-intrinsic sigmas | None) of a solved graph.
    An indeterminate system gives marg None, NaN camera sigmas and no IMU sigmas."""
    try:
        marg = gtsam.Marginals(graph, res)
        sigma_Cs = [np.sqrt(np.diag(marg.marginalCovariance(_Cc(c))))
                    for c in range(n_cams)]
        sigma_I = (np.sqrt(np.diag(marg.marginalCovariance(I0)))
                   if cfg.imu_intrinsics != "pinned" else None)
    except RuntimeError:                      # indeterminate -> report, no sigmas
        marg = None
        sigma_Cs = [np.full(calib_bases[c].DIM, np.nan) for c in range(n_cams)]
        sigma_I = None
    return marg, sigma_Cs, sigma_I


def _tga_diagnostics(diagnostics, res, marg):
    """Add the base IMU's estimated t_ga and its marginal sigma (None when unavailable)
    to `diagnostics` when the solve estimated one."""
    if res.exists(O0):
        diagnostics['t_ga'] = float(res.atVector(O0)[0])
        try:
            diagnostics['t_ga_sigma'] = float(
                np.sqrt(marg.marginalCovariance(O0)[0, 0]))
        except Exception:
            diagnostics['t_ga_sigma'] = None


def _bias_fields(res, n_states) -> dict:
    """CamImuCalibResult bias fields: the first state's biases and their drift to the last."""
    vb0 = res.atVector(_Vk(0))
    vbN = res.atVector(_Vk(n_states - 1))
    return dict(b_g=np.asarray(vb0[3:6], float), b_a=np.asarray(vb0[6:9], float),
                b_g_drift=np.asarray(vbN[3:6] - vb0[3:6], float),
                b_a_drift=np.asarray(vbN[6:9] - vb0[6:9], float))


def _aux_imu_entries(res, marg, aux_data, aux_factors) -> list:
    """CamImuCalibResult.imus: per aux IMU its solved T_IstoI, td, first biases,
    marginal sigmas, whitened chi2/dof and own t_ga (when estimated)."""
    imus = []
    for s, aux in enumerate(aux_data):
        sd = aux['seed']
        xi = np.asarray(res.atVector(_As(s)), float)
        T_sol = sd.T_IstoI.retract(xi[0:6])
        if aux.get('gyro_only', False):
            vbs0 = np.concatenate(
                [np.asarray(res.atVector(_BGs(s, 0)), float), np.zeros(3)])
            errs = [f.whitened_error(
                res.atPose3(X(ka)), res.atPose3(X(kb)),
                res.atVector(_BGs(s, ka)), res.atVector(_BGs(s, kb)),
                res.atVector(_As(s))) for ka, kb, f in aux_factors[s]]
            chi2 = float(np.mean([e @ e for e in errs]) / 6.0)
        else:
            vbs0 = np.asarray(res.atVector(_VBs(s, 0)), float)
            tga_s = (float(res.atVector(_Os(s))[0])
                     if res.exists(_Os(s)) else 0.0)
            errs = [f.whitened_error(
                res.atPose3(X(ka)), res.atPose3(X(kb)),
                res.atVector(_Vk(ka)), res.atVector(_Vk(kb)),
                res.atVector(_VBs(s, ka)), res.atVector(_VBs(s, kb)),
                res.atVector(_As(s)), tga_s) for ka, kb, f in aux_factors[s]]
            chi2 = float(np.mean([e @ e for e in errs]) / 15.0)
        try:
            sigma_A = (np.sqrt(np.diag(marg.marginalCovariance(_As(s))))
                       if marg is not None else np.full(31, np.nan))
        except RuntimeError:
            sigma_A = np.full(31, np.nan)
        entry = {
            'name': aux.get('name', f'imu{s + 1}'),
            'T_IstoI': T_sol, 'td': float(sd.td + xi[6]),
            'b_g_s': vbs0[0:3], 'b_a_s': vbs0[3:6],
            'calib_sigma': sigma_A, 'chi2_dof': chi2,
            'intr_xi24': xi[7:31],
            'gyro_only': bool(aux.get('gyro_only', False)),
            'seed_report': sd.report,
        }
        if res.exists(_Os(s)):           # aux IMU's own t_ga (+ marginal sigma)
            entry['t_ga'] = float(res.atVector(_Os(s))[0])
            try:
                entry['t_ga_sigma'] = (
                    float(np.sqrt(marg.marginalCovariance(_Os(s))[0, 0]))
                    if marg is not None else None)
            except RuntimeError:
                entry['t_ga_sigma'] = None
        imus.append(entry)
    return imus


def run_cam_imu_calibration_multi(cams, board_points, t_imu, w_m, a_m, *,
                                  aux_imus=None,
                                  config: CamImuCalibConfig | None = None,
                                  seed: CamImuSeed | None = None):
    """Joint calibration of N cameras + one IMU; see
    `_run_cam_imu_calibration_multi_once` for the model.

    Cameras flagged `retime_sensor_clock` (stamps not on the sensor clock, e.g.
    a USB camera polled at a fixed rate) are solved in two passes: pass 1
    estimates a capture-time offset per frame of those cameras
    (`solve.sensor_clock_frame_dt_sigma`, every frame kept); their capture
    times are fitted to a sensor lattice phi + n*T_s (`sensor_clock.py`); if the
    lattice passes its checks the frames are re-stamped onto it and pass 2 is
    the normal solve. A camera whose lattice fails is left as it was (reason in
    `diagnostics['sensor_clock']`)."""
    cfg = config or CamImuCalibConfig()
    flagged = [c for c, cam in enumerate(cams) if cam.get('retime_sensor_clock', False)]
    # a camera may carry a precomputed stamp map (`retime_map`, (N, 2) [t_stamp,
    # t_capture] from an earlier run's sensor-clock sidecar): it is re-stamped
    # directly and needs no pass 1.
    mapped = [c for c, cam in enumerate(cams) if cam.get('retime_map') is not None]
    if not flagged and not mapped:
        return _run_cam_imu_calibration_multi_once(cams, board_points, t_imu, w_m, a_m,
                                                   aux_imus=aux_imus, config=cfg, seed=seed)
    import dataclasses
    from .sensor_clock import fit_sensor_clock, check_sensor_clock
    for c in set(flagged) | set(mapped):
        if _cam_vision_pose(cams[c], cfg) != "imu":
            raise ValueError("retime_sensor_clock / retime_map need the IMU-propagated "
                             f"vision factor on camera {c} (solve.vision_pose: imu or "
                             "camera_input.cameras[i].vision_pose: imu)")
    report, cams2 = {}, [dict(cam) for cam in cams]
    for c in mapped:
        m = np.asarray(cams[c]['retime_map'], float).reshape(-1, 2)
        n = _apply_retime_map(cams2[c], cams[c], m[:, 0], m[:, 1])
        report[c] = {'retimed': True, 'source': 'retime_map', 'n_retimed': n,
                     'n_frames': len(cams[c]['t_cam'])}
        logger.info("camera %d: re-stamped %d / %d frames from a retime map", c, n,
                    len(cams[c]['t_cam']))
    need_pass1 = [c for c in flagged if c not in mapped]
    if need_pass1:
        sigma = float(getattr(cfg, "sensor_clock_frame_dt_sigma", 0.02))
        cams1 = []
        for c, cam in enumerate(cams):
            cam1 = dict(cams2[c] if c in mapped else cam, frame_dt=c in need_pass1)
            if c in need_pass1:
                cam1['factor_hz'] = 1e6          # every frame gets its offset
            cams1.append(cam1)
        cfg1 = dataclasses.replace(cfg, vision_imu_frame_dt_sigma=sigma)
        r1 = _run_cam_imu_calibration_multi_once(cams1, board_points, t_imu, w_m, a_m,
                                                 aux_imus=aux_imus, config=cfg1, seed=seed)
        series = np.asarray((r1.diagnostics.get('vision_imu') or {}).get('frame_dt_series', []),
                            float).reshape(-1, 3)
        for c in need_pass1:
            sel = series[series[:, 1] == c]
            if len(sel) < 10:
                report[c] = {'retimed': False, 'reason': 'too few frames with an offset'}
                continue
            t_f, d_f = sel[:, 0], sel[:, 2] * 1e-3
            clock = fit_sensor_clock(t_f + d_f)
            reasons = check_sensor_clock(clock, t_f)
            report[c] = dict(clock.report(), retimed=not reasons, reasons=reasons,
                             pass1_offset_std_ms=float(np.std(d_f) * 1e3))
            if reasons:
                logger.warning("camera %d: sensor clock not used (%s)", c, "; ".join(reasons))
                continue
            # stamp -> lattice capture time, inlier frames only (outliers are dropped)
            ok = np.asarray(clock.inlier, bool)
            t_cap = clock.capture_times()
            n = _apply_retime_map(cams2[c], cams[c], t_f[ok], t_cap[ok])
            # the map is what a corrected bag / a later solve re-applies (CLI sidecar)
            report[c]['retime_map'] = np.column_stack([t_f[ok], t_cap[ok]]).tolist()
            report[c]['dropped_stamps'] = t_f[~ok].tolist()
            report[c]['n_retimed'] = n
            report[c]['n_frames'] = len(cams[c]['t_cam'])
            logger.info("camera %d: retimed %d frames onto a %.4f Hz sensor clock "
                        "(coherence %.2f, scatter %.2f ms)", c, n, 1.0 / clock.period,
                        clock.coherence, clock.scatter * 1e3)
    dump = os.environ.get("MVIS_SENSOR_CLOCK_DUMP")
    if dump and any('retime_map' in v for v in report.values()):
        # crash-safe copy of the fitted stamp maps (pass 1 is the expensive part):
        # same layout as the CLI sidecar, so `retime_map:` can consume it directly
        from mvis.io.sensor_clock import camera_record, write_sidecar
        write_sidecar(dump, {str(c): camera_record(v) for c, v in report.items() if 'retime_map' in v})
        logger.info("sensor-clock stamp maps dumped to %s", dump)
    r2 = _run_cam_imu_calibration_multi_once(cams2, board_points, t_imu, w_m, a_m,
                                             aux_imus=aux_imus, config=cfg, seed=seed)
    r2.diagnostics['sensor_clock'] = {int(c): v for c, v in report.items()}
    if need_pass1 and (r1.diagnostics.get('lm_no_step') or {}).get('failed'):
        # the offsets are all zero, so a lattice fitted to them says nothing about the sensor clock
        r2.diagnostics['sensor_clock_pass1_lm_no_step'] = r1.diagnostics['lm_no_step']
        r2.ok = False
    return r2


def _apply_retime_map(cam2, cam, t_stamp, t_capture) -> int:
    """Re-stamp `cam`'s frames in place into `cam2` from a stamp -> capture-time map.

    Frames whose stamp (rounded to 1 µs) is not in the map are dropped; the kept
    frames are re-sorted by their new times and the per-frame lists re-aligned.
    The vision-factor rate is NOT touched: pass 2 / a mapped solve honours the
    configured `factor_hz` like any other camera (2026-10-31; the earlier
    behaviour forced every frame). Returns the number of frames kept."""
    new_t = {round(float(t), 6): float(tc) for t, tc in zip(t_stamp, t_capture)}
    keep = [i for i, t in enumerate(cam['t_cam']) if round(float(t), 6) in new_t]
    t_new = np.asarray([new_t[round(float(cam['t_cam'][i]), 6)] for i in keep], float)
    o = np.argsort(t_new)
    cam2['t_cam'] = t_new[o]
    cam2['ids_list'] = [cam['ids_list'][keep[i]] for i in o]
    cam2['uv_list'] = [cam['uv_list'][keep[i]] for i in o]
    cam2['frames'] = None                          # stamps changed: ragged arrays only
    cam2.pop('retime_map', None)
    cam2['retime_sensor_clock'] = False
    return len(keep)


def _run_cam_imu_calibration_multi_once(cams, board_points, t_imu, w_m, a_m, *,
                                        aux_imus=None,
                                        config: CamImuCalibConfig | None = None,
                                        seed: CamImuSeed | None = None):
    """Joint calibration of N cameras + one IMU. `cams` is a list of
    per-camera dicts {t_cam, ids_list, uv_list, cal_vec, is_fisheye,
    image_size}; cams[0] is the BASE camera (defines the state clock and the
    world seed). Every camera gets its own CameraCalib bundle; every frame
    (base included) is constrained by its two bracketing states through
    CameraCalibInterpFactor (solve.vision_pose "interp"), or by one anchor
    state propagated through the IMU (CameraCalibImuFactor, "imu").

    `seed` may be passed to skip the base init (e.g. a window-selected seed
    when the whole-sequence init gated ok=False)."""
    cfg = config or CamImuCalibConfig()
    t_imu, w_m, a_m, aux_imus = _upsampled_imus(cfg, t_imu, w_m, a_m, aux_imus)
    board_points = np.asarray(board_points, float)
    n_cams = len(cams)
    timings = {}
    _tic = time.time()
    logger.info("cam-IMU: initialising %d camera(s), %d frames, %d IMU samples",
                n_cams, sum(len(c["t_cam"]) for c in cams), len(t_imu))

    # --- per-camera PnP track + init; base supplies world/states/board ---
    tracks, seeds = [], []
    for c, cam in enumerate(cams):
        cal_vec = np.asarray(cam['cal_vec'], float)
        model = _cam_model(cam)
        t_kept, T_CtoB_kept, _n = pnp_pose_track(
            cam['t_cam'], cam['ids_list'], cam['uv_list'], board_points,
            cal_vec, model=model,
            pnp_method=getattr(cfg, 'pnp_method', 'iterative'))
        tracks.append((t_kept, T_CtoB_kept))
        if c == 0 and seed is not None:
            seeds.append(seed)
        else:
            seeds.append(initialize_cam_imu(t_kept, T_CtoB_kept,
                                            t_imu, w_m, a_m, g_mag=cfg.gravity,
                                            **cfg.init_kwargs))
    seed = seeds[0]

    # extra-camera consistency gates vs the base seed. Each init's gravity
    # world has ARBITRARY YAW, so T_BinW is only gauge-comparable — the
    # yaw-invariant check is the gravity direction expressed in the SHARED
    # physical board frame: both cameras observed the same board hanging in
    # the same gravity field, so g_inB must agree.
    board_agreement = []
    g0 = np.asarray(seed.g_inB, float)
    g0 = g0 / np.linalg.norm(g0)
    for c in range(1, n_cams):
        gc = np.asarray(seeds[c].g_inB, float)
        gc = gc / np.linalg.norm(gc)
        board_agreement.append(
            {'g_dir_deg': float(np.rad2deg(np.arccos(
                np.clip(g0 @ gc, -1.0, 1.0)))),
             'g_norm_ls': seeds[c].report.get('g_norm_ls')})

    # --- states: base camera accepted frames, decimated ---
    t_kept0, T_CtoB_kept0 = tracks[0]
    keep = _decimate(t_kept0, cfg.state_hz)
    t_states = np.asarray(t_kept0, float)[keep]
    T_CtoB_states = [T for T, kp in zip(T_CtoB_kept0, keep) if kp]

    g_inW = np.array([0.0, 0.0, -cfg.gravity])
    T_CtoI_inv = seed.T_CtoI.inverse()
    # T_ItoW = T_BinW . T_CtoB . T_CtoI^-1  (camera-to-world via the board)
    T_init = [seed.T_BinW.compose(T_CtoB).compose(T_CtoI_inv)
              for T_CtoB in T_CtoB_states]
    t_states, T_init = _subdivide_states(t_states, T_init,
                                         getattr(cfg, "knot_hz", 0.0),
                                         cfg.max_interp_dt)
    v_init = _densify_velocities(t_states, T_init, seed, t_imu, w_m, a_m,
                                 cfg, g_inW)

    # --- per-camera frames -> bracketing states ---
    calib_bases, image_sizes, frames_per_cam, n_dropped = [], [], [], []
    frame_fates = []
    pids_per_cam = []                     # corner board-point ids, parallel to frames
    frame_selection = []                # per-cam {timestamp: "stage:reason"}
    for c, cam in enumerate(cams):
        cal_vec = np.asarray(cam['cal_vec'], float)
        model = _cam_model(cam)
        calib_bases.append(CameraCalib(seeds[c].T_CtoI, seeds[c].td, 0.0,
                                       cal_vec, model=model))
        image_sizes.append(cam['image_size'])
        t_kept_c = np.asarray(tracks[c][0], float)
        keep_c = _decimate(t_kept_c, _cam_factor_hz(cam, cfg))
        t_frames = t_kept_c[keep_c]
        j, dt_frame, ok_f = _assign_brackets(t_frames, seeds[c].td, t_states,
                                             cfg.max_interp_dt)
        idx_by_time = {float(t): i
                       for i, t in enumerate(np.asarray(cam['t_cam'], float))}
        # 2c-15: consume the REAL FrameObservations when the loader provides
        # them (first seam off the ragged tuples); the ragged arrays remain
        # the fallback for external ragged-only callers, and while both are
        # present they are asserted byte-equal (they derive from the same
        # frames via frames_to_arrays — this guards against drift).
        fo_by_time = ({float(f.timestamp): f for f in cam["frames"]}
                      if cam.get("frames") else None)
        frames = []
        frame_pids = []
        used_ts = set()
        tr_hint = float(cam.get('tr_hint', 0.0) or 0.0)
        split_rows = (getattr(cfg, "knot_hz", 0.0) > 0 and tr_hint > 0
                      and not _cam_pin_tr(cam, cfg))
        h_px = int(cam['image_size'][1])
        rmode = int(cam.get('rolling_mode', cfg.rolling_mode))
        for m, t in enumerate(t_frames):
            if not ok_f[m]:
                continue
            i = idx_by_time[float(t)]
            if fo_by_time is not None:
                fo = fo_by_time[float(t)]
                ids = np.array([o.point_id for o in fo.observations], int)
                uv = np.array([o.pixel for o in fo.observations],
                              float).reshape(-1, 2)
                assert ids.tobytes() == np.asarray(
                    cam['ids_list'][i], int).tobytes()
                assert uv.tobytes() == np.asarray(
                    cam['uv_list'][i], float).reshape(-1, 2).tobytes()
            else:
                ids = np.asarray(cam['ids_list'][i], int)
                uv = np.asarray(cam['uv_list'][i], float).reshape(-1, 2)
            cm = corner_mask(ids, cfg.corner_subsample)
            if np.sum(cm) < 6:                # keep PnP-grade frames only
                continue
            if split_rows:
                # dense knots + rolling shutter: the readout spans several knot
                # intervals, so give each corner the bracket of ITS row's capture
                # time (seed td + rho * tr_hint) instead of extrapolating one
                # bracket across the readout. One sub-frame factor per bracket.
                uv_c, pts_c = uv[cm], board_points[ids[cm]]
                rho = np.array([row_offset(v, h_px, rmode) for v in uv_c[:, 1]])
                jj = np.clip(np.searchsorted(
                    t_states, float(t) + seeds[c].td + rho * tr_hint),
                    1, len(t_states) - 1)
                for jg in np.unique(jj):
                    g = jj == jg
                    frames.append((uv_c[g], pts_c[g], int(jg),
                                   float(t) - float(t_states[jg - 1])))
                    frame_pids.append(ids[cm][g])
            else:
                frames.append((uv[cm], board_points[ids[cm]],
                               int(j[m]), float(dt_frame[m])))
                frame_pids.append(ids[cm])
            used_ts.add(float(t))
        frames_per_cam.append(frames)
        pids_per_cam.append(frame_pids)
        n_dropped.append(int(np.sum(~ok_f)))
        frame_fates.append(_frame_fate(
            cam, set(np.asarray(cam['t_cam'], float).tolist())
            - set(t_kept_c.tolist()) | set(t_frames.tolist()), used_ts))
        # 2c-12: the selection story, per frame, at the moment of selection —
        # first failing filter wins (the later ones never saw the frame)
        _kept = set(float(x) for x in t_kept_c)
        _dec = set(float(x) for x in t_frames)
        _ok = {float(t_frames[m]) for m in range(len(t_frames)) if ok_f[m]}
        _sel = {}
        for _t in np.asarray(cam['t_cam'], float):
            _t = float(_t)
            if _t in used_ts:
                continue
            if _t not in _kept:
                _sel[_t] = "camimu:pnp_reject"
            elif _t not in _dec:
                _sel[_t] = "camimu:factor_hz"
            elif _t not in _ok:
                _sel[_t] = "camimu:bracket_unassigned"
            else:
                _sel[_t] = "camimu:min_corners"
        frame_selection.append(_sel)

    timings['init_s'] = time.time() - _tic
    _tic = time.time()

    # --- aux IMUs: I6 init per extra IMU against the base gyro ---
    aux_data = []
    for aux in (aux_imus or []):
        sd = initialize_imu_imu(t_imu, w_m, aux['t_imu'], aux['w_m'],
                                b_g_base=seed.b_g)
        aux_data.append({**aux, 'seed': sd})

    timings['aux_init_s'] = time.time() - _tic
    _tic = time.time()
    logger.info("cam-IMU: init done in %.1f s; building the factor graph",
                timings['init_s'] + timings['aux_init_s'])

    rolling_modes = [int(cam.get('rolling_mode', cfg.rolling_mode))
                     for cam in cams]
    pin_trs = [_cam_pin_tr(cam, cfg) for cam in cams]
    pin_intr = [bool(cam.get('pin_intrinsics', False)) for cam in cams]
    pin_extrinsics = [_cam_pin_extr(cam, cfg) for cam in cams]
    frame_dt_cams = [bool(cam.get('frame_dt', True)) for cam in cams]
    vision_pose_cams = [_cam_vision_pose(cam, cfg) for cam in cams]
    graph, values, cam_factors, imu_factors, aux_factors = build_joint_graph(
        t_states, frames_per_cam, T_init, v_init, seed, calib_bases,
        image_sizes, t_imu, w_m, a_m, cfg, aux_data=aux_data,
        rolling_modes=rolling_modes, pin_trs=pin_trs, pin_intr=pin_intr,
        pin_extrinsics=pin_extrinsics, frame_dt_cams=frame_dt_cams,
        board_pids_per_cam=pids_per_cam, board_points=board_points,
        vision_pose_cams=vision_pose_cams)
    timings['graph_build_s'] = time.time() - _tic

    # --- optimize ---
    perturb_rel = maybe_perturb_initial_values(values)   # MVIS_PERTURB_INIT (debug)
    _tic = time.time()
    rms_pre, _ = _reproj_rms(cam_factors, values)      # one corner pass at `values`
    timings['pre_diag_s'] = time.time() - _tic
    _tic = time.time()
    params = _lm_params(cfg)
    tga_relin_s = float(getattr(cfg, "tga_relin_threshold_s", 0.0) or 0.0)
    if tga_relin_s > 0.0 and est_t_ga_run(cfg):
        streams = _tga_streams(cfg, t_states, t_imu, w_m, a_m, imu_factors,
                               aux_data, aux_factors, cam_factors)

        def make_opt(start, lm_params):
            return TgaRelinOptimizer(graph, start, lm_params, streams, tga_relin_s)
    else:
        def make_opt(start, lm_params):
            return gtsam.LevenbergMarquardtOptimizer(graph, start, lm_params)
    opt = make_opt(values, params)
    finite_pre = rms_pre[np.isfinite(rms_pre)]          # per-frame RMS; all-guarded frames are NaN
    logger.info("cam-IMU: graph built in %.1f s (%d camera factors, %d IMU factors); "
                "optimising: error %.6g, median per-frame reprojection RMS %.3f px, "
                "max %d iterations",
                timings['graph_build_s'], len(cam_factors), len(imu_factors), opt.error(),
                float(np.median(finite_pre)) if finite_pre.size else float("nan"),
                params.getMaxIterations())
    # The optimizer already evaluates graph.error at the initial Values
    # (constructor) and at every accepted iterate (tryLambda), so its state
    # error IS graph.error(values) / graph.error(res) — no extra error passes
    # (tests/cam_imu/test_diagnostics_once.py pins the bitwise equality).
    err_init = opt.error()
    res = optimize_logged(opt, params, "cam-IMU LM")
    res, opt, lm_rescue = rescue_if_stuck(graph, values, res, opt, params, make_opt)
    err_final = opt.error()
    timings['solve_s'] = time.time() - _tic
    lm_no_step = _lm_no_step_report(graph, values, opt, cfg)
    _tic = time.time()

    # --- diagnostics --- one corner-norm table at `res` feeds the per-frame
    # RMS series AND the pooled Kalibr population (was two identical passes)
    table_post = _corner_norm_table(cam_factors, res)
    rms_post, fac_cam = _reproj_rms(cam_factors, res, table=table_post)
    corner_px, corner_cam = _reproj_corner_norms(cam_factors, res, table=table_post)
    imu_split, imu_per_interval = _imu_cost_split(imu_factors, res, O0 if est_t_ga_run(cfg) else None)
    imu_blocks = guarded(lambda: _imu_block_residuals(imu_factors, res, O0 if est_t_ga_run(cfg) else None))
    timings['post_diag_s'] = time.time() - _tic
    n_cam_dof = int(sum(2 * f.n for _c, _j, f in cam_factors))
    n_imu_dof = 15 * len(imu_factors)

    _tic = time.time()
    logger.info("cam-IMU: solve done in %.1f s (%d iterations, error %.6g -> %.6g); "
                "computing uncertainties", timings['solve_s'], opt.iterations(),
                err_init, err_final)
    marg, sigma_Cs, sigma_I = _marginal_sigmas(graph, res, calib_bases, n_cams, cfg)

    timings['marginals_s'] = time.time() - _tic
    timings['per_iteration_s'] = (timings['solve_s']
                                  / max(opt.iterations(), 1))
    logger.info("stage timings: init %.1f s, aux_init %.1f s, graph_build %.1f s, "
                "pre_diag %.1f s, solve %.1f s (%d iterations, %.1f s/it), "
                "post_diag %.1f s, marginals %.1f s",
                timings['init_s'], timings['aux_init_s'], timings['graph_build_s'],
                timings['pre_diag_s'], timings['solve_s'], opt.iterations(),
                timings['per_iteration_s'], timings['post_diag_s'],
                timings['marginals_s'])
    sols = [calib_bases[c].retract(res.atVector(_Cc(c)))
            for c in range(n_cams)]
    accounting = _guarded_accounting(n_cams, lambda: _corner_accounting(
        cam_factors, res, sols, image_sizes, frame_fates))
    cameras = []
    for c in range(n_cams):
        m = fac_cam == c
        entry = {
            'calib': sols[c], 'calib_sigma': sigma_Cs[c],
            **reproj_entry(corner_px[corner_cam == c], cfg.sigma_px),
            'n_frames': int(np.sum(m)), 'n_dropped': n_dropped[c],
            'image_size': image_sizes[c],
            'corner_accounting': accounting[c],
        }
        if c > 0:
            # rig extrinsic camera_c -> camera_0, first-order sigma from the
            # joint xi marginal: T_rel' = T_rel Exp(-Ad(T_rel^-1) d0 + dc)
            T_rel = sols[0].T.inverse().compose(sols[c].T)
            entry['T_CtoC0'] = T_rel
            if marg is not None:
                kv = gtsam.KeyVector([_Cc(0), _Cc(c)])
                P = marg.jointMarginalCovariance(kv).fullMatrix()
                # joint cov of [ξ_0, ξ_c]; cam_c's block starts at cam0's DIM
                # (each cam's pose is the first 6 of its ξ). Handles mixed models.
                off = calib_bases[0].DIM
                ps, pe = off, off + 6
                J = np.hstack([-T_rel.inverse().AdjointMap(), np.eye(6)])
                P6 = np.zeros((12, 12))
                P6[0:6, 0:6] = P[0:6, 0:6]
                P6[0:6, 6:12] = P[0:6, ps:pe]
                P6[6:12, 0:6] = P[ps:pe, 0:6]
                P6[6:12, 6:12] = P[ps:pe, ps:pe]
                entry['T_CtoC0_sigma'] = np.sqrt(np.diag(J @ P6 @ J.T))
            entry['seed_gates'] = board_agreement[c - 1]
        _note_kb_domain(entry, c)
        cameras.append(entry)

    imus = _aux_imu_entries(res, marg, aux_data, aux_factors)

    # guarded corners are zero rows in the optimizer's residual, so the sum of
    # the valid corner norms squared IS the total squared vision residual
    cam_cost = float(np.sum(corner_px ** 2)) / cfg.sigma_px ** 2 * 0.5
    diagnostics = {
        'gravity': cfg.gravity, 'gravity_source': cfg.gravity_source, 'gravity_site': cfg.gravity_site,
        'gravity_place': cfg.gravity_place,
        'iterations': opt.iterations(),
        'tga_relinearizations': list(getattr(opt, 'relinearizations', [])),
        'err_init': err_init, 'err_final': err_final,
        'reproj_rms_pre_px': rms_pre, 'reproj_rms_post_px': rms_post,
        **reproj_entry(corner_px, cfg.sigma_px),
        'reproj_frame_cam': fac_cam,
        # per camera: signed corner errors by frame + norms by board corner (report's scatter
        # and corner heatmap); in memory only, cam_imu.json keeps the summary
        'reproj_corners_by_cam': table_post.by_camera(cam_factors),
        'imu_cost_split': imu_split,
        'imu_interval_cost': imu_per_interval,
        'imu_block_residuals': imu_blocks,     # per interval, per state block (report section IMU Residuals)
        'imu_chi2_dof': float(sum(imu_split.values()) / max(n_imu_dof, 1)),
        'cam_chi2_dof': float(2 * cam_cost / max(n_cam_dof, 1)),
        'shift_from_seed_xi': np.asarray(res.atVector(C0), float),
        'n_states': len(t_states),
        'n_factors': {'vision': len(cam_factors), 'imu': len(imu_factors),
                      'aux_imu': sum(len(f) for f in aux_factors)},
        'vision_pose': (vision_pose_cams[0] if len(set(vision_pose_cams)) == 1
                        else list(vision_pose_cams)),
        'vision_imu': (_vision_imu_diagnostics(cam_factors, sols, t_states, marg,
                                               _imu_model(cfg), res)
                       if "imu" in vision_pose_cams else None),
        'timings': timings,
        'seed_report': seed.report,
        'board_agreement': board_agreement,
    }
    if lm_no_step is not None:
        diagnostics['lm_no_step'] = lm_no_step
    if lm_rescue is not None:
        diagnostics['lm_rescue'] = lm_rescue
    if res.exists(H0):
        diagnostics['board_heights'] = _board_heights_report(
            res, marg, board_points, pids_per_cam)
    _tga_diagnostics(diagnostics, res, marg)
    # init gates (g_norm_ok, scale_ok) judge the seed, not the solve: a raw IMU
    # with a large accel offset (TUM-VI BMI160, ~1.3 m/s²) fails |g| at init and
    # still solves well. Warn and keep them in diagnostics['seed_report'].
    failed_gates = [g for g in ('g_norm_ok', 'scale_ok') if seed.report.get(g) is False]
    if failed_gates:
        logger.warning("cam-IMU init gate(s) failed: %s (|g| %.3f, scale %.3f); "
                       "seed only, not used for ok", ', '.join(failed_gates),
                       seed.report.get('g_norm_ls', float('nan')),
                       seed.report.get('scale', float('nan')))
    ok = bool(np.isfinite(diagnostics['reproj_rms_px']))
    rms_gate = float(getattr(cfg, 'rms_gate_px', 0.0) or 0.0)
    if rms_gate > 0 and diagnostics['reproj_rms_px'] > rms_gate:
        ok = False
    if lm_no_step is not None and lm_no_step['failed']:
        ok = False
    if lm_rescue is not None and lm_rescue['failed']:
        ok = False

    # MVIS_DUMP_STATE (debug): full-precision npz for the speed-up gates
    maybe_dump_state(
        res,
        calibs={f"c{c}": sols[c] for c in range(n_cams)},
        sigmas={**{f"c{c}": sigma_Cs[c] for c in range(n_cams)}, "I0": sigma_I,
                **{f"A{s}": e['calib_sigma'] for s, e in enumerate(imus)},
                "o0": (np.array([diagnostics['t_ga_sigma']])
                       if diagnostics.get('t_ga_sigma') is not None else None)},
        scalars={'err_init': float(err_init), 'err_final': float(err_final),
                 'iterations': int(opt.iterations()), 'n_states': len(t_states),
                 'cam_chi2_dof': diagnostics['cam_chi2_dof'],
                 'imu_chi2_dof': diagnostics['imu_chi2_dof'],
                 'ok': bool(ok),
                 'perturb_init_rel': (0.0 if perturb_rel is None else perturb_rel)},
        arrays={'rms_pre': rms_pre, 'rms_post': rms_post, 'fac_cam': fac_cam,
                'corner_px': corner_px, 'corner_cam': corner_cam,
                'imu_interval_cost': imu_per_interval, 'imu_cost_split': imu_split,
                't_states': t_states},
        timings=timings)

    return CamImuCalibResult(
        calib=sols[0], calib_sigma=sigma_Cs[0],
        frame_selection=frame_selection,
        **_bias_fields(res, len(t_states)),
        imu_intr_xi=np.asarray(res.atVector(I0), float),
        imu_model=imu_model_name(_imu_model(cfg)),
        imu_intr_sigma=sigma_I,
        T_BinW=res.atPose3(B0), seed=seed, values=res,
        t_states=t_states, diagnostics=diagnostics, ok=ok,
        cameras=cameras, imus=imus)


def run_cam_imu_calibration(t_cam, ids_list, uv_list, board_points, cal_vec,
                            t_imu, w_m, a_m, image_size, *,
                            is_fisheye: bool = False,
                            model: str | None = None,
                            config: CamImuCalibConfig | None = None,
                            seed: CamImuSeed | None = None):
    """Single-camera entry: delegates to the multi-camera path with one cam."""
    return run_cam_imu_calibration_multi(
        [{'t_cam': t_cam, 'ids_list': ids_list, 'uv_list': uv_list,
          'cal_vec': cal_vec, 'is_fisheye': is_fisheye, 'model': model,
          'image_size': image_size}],
        board_points, t_imu, w_m, a_m, config=config, seed=seed)


def run_cam_imu_calibration_feature(t_cam, ids_list, uv_list, cal_vec,
                                    t_imu, w_m, a_m, image_size, *,
                                    is_fisheye: bool = False,
                                    model: str | None = None,
                                    sfm_mode: str = "general",
                                    sfm_kwargs: dict | None = None,
                                    config: CamImuCalibConfig | None = None,
                                    pin_intrinsics: bool = True,
                                    td_pin: float | None = None):
    """Single-camera FEATURE-MODE joint cam-IMU calibration: one camera through
    `run_cam_imu_calibration_feature_multi`.

    The vision factors run at `config.state_hz`, so the frames sit at the states
    (the interpolation evaluates near an interval end, which keeps the td chord
    error small). `td_pin`: start td at this exact value instead of the seed's
    (with `config.sigma_td_prior` > 0 it is held there)."""
    cfg = config or CamImuCalibConfig()
    cam = {'t_cam': t_cam, 'ids_list': ids_list, 'uv_list': uv_list,
           'cal_vec': cal_vec, 'is_fisheye': is_fisheye, 'model': model,
           'image_size': image_size, 'factor_hz': cfg.state_hz, 'td_pin': td_pin}
    return run_cam_imu_calibration_feature_multi(
        [cam], t_imu, w_m, a_m, config=cfg, pin_intrinsics=pin_intrinsics,
        sfm_mode=sfm_mode, sfm_kwargs=sfm_kwargs)


def run_cam_imu_calibration_feature_multi(cams, t_imu, w_m, a_m, *,
                                          aux_imus=None,
                                          config: CamImuCalibConfig | None = None,
                                          pin_intrinsics: bool = True,
                                          sfm_mode: str = "general",
                                          sfm_kwargs: dict | None = None):
    """Multi-camera + multi-IMU FEATURE-MODE joint calibration.

    `cams[0]` is the BASE camera (defines the world/states/landmark map). Each
    cam dict: {t_cam, ids_list, uv_list, cal_vec, is_fisheye, image_size}
    (+ optional rolling_mode, pin_tr, pin_intrinsics, factor_hz, td_pin: start
    that camera's td at this value instead of its seed's). All cameras share the
    SAME world landmarks keyed by global tag id (ArUco ids are unique across
    cameras), each reprojected through its own `T_CtoI_c`. `aux_imus`: list of
    {name, t_imu, w_m, a_m, gyro_only?} calibrated against the base gyro (I6).
    Camera intrinsics pinned from the prior calibration by default."""
    from .mono_sfm import normalize_detections, planar_sfm, general_sfm
    cfg = config or CamImuCalibConfig()
    t_imu, w_m, a_m, aux_imus = _upsampled_imus(cfg, t_imu, w_m, a_m, aux_imus)
    g_inW = np.array([0.0, 0.0, -cfg.gravity])
    sfm_fn = general_sfm if sfm_mode == "general" else planar_sfm
    sfm_kwargs = sfm_kwargs or {}

    def _sfm_seed(cam):
        tr = normalize_detections(cam['ids_list'], cam['uv_list'],
                                  np.asarray(cam['cal_vec'], float),
                                  bool(cam.get('is_fisheye', False)),
                                  model=_cam_model(cam))
        sfm = sfm_fn(cam['t_cam'], [t[0] for t in tr], [t[1] for t in tr],
                     **sfm_kwargs)
        seed = initialize_cam_imu(sfm.t, sfm.poses, t_imu, w_m, a_m,
                                  board_mode=False, g_mag=cfg.gravity, **cfg.init_kwargs)
        return sfm, seed

    # --- base camera: trajectory + shared metric world landmark map ---
    sfm0, seed0 = _sfm_seed(cams[0])
    R_BtoW = np.asarray(seed0.R_BtoW, float)
    s = float(seed0.s)
    T_CtoI0_inv = seed0.T_CtoI.inverse()
    T_ItoW_dense = []
    for P in sfm0.poses:
        R = R_BtoW @ P.rotation().matrix()
        p = R_BtoW @ (s * np.asarray(P.translation(), float))
        T_ItoW_dense.append(
            gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(*p)).compose(T_CtoI0_inv))

    def _world_pt(tid):
        return R_BtoW @ (s * np.asarray(sfm0.points[tid], float))

    keep = _decimate(np.asarray(sfm0.t, float), cfg.state_hz)
    t_states = np.asarray(sfm0.t, float)[keep]
    T_init = [T for T, k in zip(T_ItoW_dense, keep) if k]
    v_init = _densify_velocities(t_states, T_init, seed0, t_imu, w_m, a_m,
                                 cfg, g_inW)

    # --- per-camera extrinsic seed + frames over the SHARED landmark map ---
    calib_bases, image_sizes, frames_per_cam, seeds, n_dropped = [], [], [], [], []
    tried_ts, frame_time = [], []         # per camera, for the corner accounting
    used = set()
    for c, cam in enumerate(cams):
        cal_vec = np.asarray(cam['cal_vec'], float)
        fish = bool(cam.get('is_fisheye', False))
        seed_c = seed0 if c == 0 else _sfm_seed(cam)[1]   # init vs the base IMU
        seeds.append(seed_c)
        td_c = seed_c.td if cam.get('td_pin') is None else float(cam['td_pin'])
        calib_bases.append(CameraCalib(seed_c.T_CtoI, td_c, 0.0, cal_vec, fish,
                                       model=_cam_model(cam)))
        image_sizes.append(cam['image_size'])
        t_kept = np.asarray(cam['t_cam'], float)
        keep_c = _decimate(t_kept, _cam_factor_hz(cam, cfg))
        t_frames = t_kept[keep_c]
        j, dt_frame, ok_f = _assign_brackets(t_frames, td_c, t_states,
                                             cfg.max_interp_dt)
        idx_by_time = {float(t): i for i, t in enumerate(t_kept)}
        frames, drop = [], 0
        frame_time.append({})
        for m, t in enumerate(t_frames):
            if not ok_f[m]:
                drop += 1
                continue
            i = idx_by_time[float(t)]
            ids = np.asarray(cam['ids_list'][i], int)
            uv = np.asarray(cam['uv_list'][i], float).reshape(-1, 2)
            cmask = corner_mask(ids, cfg.corner_subsample)
            sel = [k for k in np.nonzero(cmask)[0] if int(ids[k]) in sfm0.points]
            if len(sel) < 6:
                continue
            fids = [int(ids[k]) for k in sel]
            frames.append((uv[sel], fids, int(j[m]), float(dt_frame[m])))
            frame_time[-1][(int(j[m]), float(dt_frame[m]))] = float(t)
            used.update(fids)
        frames_per_cam.append(frames)
        n_dropped.append(drop)
        tried_ts.append(t_frames)
    # cull under-observed landmarks (few-frame / weakly-triangulated points fly
    # away in the joint solve). Count obs across ALL cameras; keep ids seen in
    # >= min_landmark_obs frames; drop them from the frames too.
    min_obs = int(getattr(cfg, "min_landmark_obs", 0))
    if min_obs > 0:
        from collections import Counter
        cnt = Counter(fid for frames in frames_per_cam
                      for (_uv, fids, _j, _dt) in frames for fid in fids)
        keep_ids = {i for i, c in cnt.items() if c >= min_obs}
        culled = []
        for frames in frames_per_cam:
            nf = []
            for uv, fids, j, dt in frames:
                sel = [k for k, fid in enumerate(fids) if fid in keep_ids]
                if len(sel) < 6:
                    continue
                nf.append((uv[sel], [fids[k] for k in sel], j, dt))
            culled.append(nf)
        frames_per_cam = culled
        used = keep_ids
    landmarks = {tid: _world_pt(tid) for tid in used}

    # --- aux IMUs (I6, vs the base gyro) ---
    aux_data = []
    for aux in (aux_imus or []):
        sd = initialize_imu_imu(t_imu, w_m, aux['t_imu'], aux['w_m'],
                                b_g_base=seed0.b_g)
        aux_data.append({**aux, 'seed': sd})

    pin_intr = [bool(cam.get('pin_intrinsics', pin_intrinsics)) for cam in cams]
    rolling_modes = [int(cam.get('rolling_mode', cfg.rolling_mode)) for cam in cams]
    pin_trs = [_cam_pin_tr(cam, cfg) for cam in cams]
    pin_extrinsics = [_cam_pin_extr(cam, cfg) for cam in cams]

    def _solve(graph, values):
        if getattr(cfg, "optimizer", "lm") == "dogleg":   # trust-region step cap
            p = gtsam.DoglegParams(); p.setMaxIterations(cfg.max_iterations)
            o = gtsam.DoglegOptimizer(graph, values, p)
        else:
            p = _lm_params(cfg, lambda_initial=getattr(cfg, "lm_lambda_init", 0.0))
            o = gtsam.LevenbergMarquardtOptimizer(graph, values, p)
        r = optimize_logged(o, p, "cam-IMU LM")
        return r, o.iterations()

    def _fcorner_norms(vals, cfacs):
        """(pooled per-corner norms, camera per corner, per-frame norm-RMS,
        camera per frame) — one projection pass for stats AND the plot."""
        out = np.empty(len(cfacs)); fc = np.empty(len(cfacs), int)
        norms, ncam = [], []
        for q, (c, _jj, fac) in enumerate(cfacs):
            e = fac.corner_norms_values(vals)
            v = np.isfinite(e)
            norms.append(e[v]); ncam.append(np.full(int(v.sum()), c, int))
            out[q] = np.sqrt(np.mean(e[v] ** 2)) if v.any() else np.nan
            fc[q] = c
        return (np.concatenate(norms) if norms else np.empty(0),
                np.concatenate(ncam) if ncam else np.empty(0, int), out, fc)

    def _freproj(vals, cfacs):
        _, _, out, fc = _fcorner_norms(vals, cfacs)
        return out, fc

    # 2-pass per-OBSERVATION outlier rejection: solve -> drop corners with
    # robust-outlier reprojection -> rebuild + re-solve (warm-started). The
    # feature graph otherwise has NO hard outlier reject (Huber only masks a few
    # grazing pinhole corners that blow up the solve — the bag-13 class). The
    # front-end (SfM/init) runs once; only the joint LM repeats, warm-started.
    reject = bool(getattr(cfg, "outlier_reject", False))   # depth guard suffices; reject is opt-in
    floor_px = float(getattr(cfg, "outlier_floor_px", 10.0))
    kmad = float(getattr(cfg, "outlier_k", 5.0))
    max_rounds = (int(getattr(cfg, "outlier_iters", 2)) + 1) if reject else 1
    Ti, vi, landmarks_init = T_init, v_init, dict(landmarks)
    drop_log, pre_per_cam = [], {}
    n_before_reject = [{(f[2], f[3]): len(f[1]) for f in fr} for fr in frames_per_cam]
    for rnd in range(max_rounds):
        graph, values, cam_factors, imu_factors, aux_factors = build_joint_graph(
            t_states, frames_per_cam, Ti, vi, seed0, calib_bases,
            image_sizes, t_imu, w_m, a_m, cfg, aux_data=aux_data,
            rolling_modes=rolling_modes, pin_trs=pin_trs, pin_intr=pin_intr,
            pin_extrinsics=pin_extrinsics,
            structure="feature", landmarks_init=landmarks_init)
        if rnd == 0:
            rp, fp = _freproj(values, cam_factors)
            pre_per_cam = {int(c): float(np.median(rp[fp == c]))
                           for c in range(len(cams)) if (fp == c).any()}
        res, n_iters = _solve(graph, values)
        if rnd == max_rounds - 1:
            break
        # per-corner residuals + robust per-camera threshold (MAD)
        facres, percam = [], {}
        for c, jj, fac in cam_factors:
            P = np.array([res.atPoint3(_L(i)) for i in fac.ids])
            rc = np.linalg.norm(fac.error(
                res.atPose3(X(jj - 1)), res.atPose3(X(jj)),
                res.atVector(_Cc(c)), P).reshape(-1, 2), axis=1)
            facres.append((c, jj, fac, rc)); percam.setdefault(c, []).append(rc)
        tau = {}
        for c, lst in percam.items():
            a = np.concatenate(lst); med = float(np.median(a))
            tau[c] = max(floor_px, kmad * 1.4826 * float(np.median(np.abs(a - med))))
        new_fpc = [[] for _ in range(len(frames_per_cam))]
        ndrop = 0
        for c, jj, fac, rc in facres:
            keep = rc <= tau[c]
            ndrop += int(fac.n - int(keep.sum()))
            if int(keep.sum()) >= 6:
                new_fpc[c].append((np.asarray(fac.uv)[keep],
                                   [int(fac.ids[k]) for k in range(fac.n) if keep[k]],
                                   jj, fac.dt_frame))
        drop_log.append(ndrop)
        if ndrop == 0:
            break
        frames_per_cam = new_fpc
        surviving = {i for cam in frames_per_cam for (uv, ids, j, dt) in cam for i in ids}
        landmarks_init = {i: np.asarray(res.atPoint3(_L(i)), float) for i in surviving}
        Ti = [res.atPose3(X(k)) for k in range(len(t_states))]
        vi = [np.asarray(res.atVector(_Vk(k))[:3], float) for k in range(len(t_states))]
    landmarks = landmarks_init
    corner_px, corner_cam, rms, fcam = _fcorner_norms(res, cam_factors)

    corner_diag = None
    if getattr(cfg, "corner_diag", False):    # per-corner (residual_px, fov_deg, depth_m) for base cam
        calib0 = calib_bases[0].retract(res.atVector(_Cc(0)))
        fx, fy, cx, cy = (float(v) for v in calib0.intr[:4])   # any model
        cd = []
        for (c, jj, fac) in cam_factors:
            if c != 0:
                continue
            P = np.array([res.atPoint3(_L(i)) for i in fac.ids])
            e = fac.error(res.atPose3(X(jj - 1)), res.atPose3(X(jj)),
                          res.atVector(_Cc(c)), P).reshape(-1, 2)
            rc = np.linalg.norm(e, axis=1)
            for k in range(fac.n):
                u, v = fac.uv[k]
                fov = np.degrees(np.arctan(np.hypot((u - cx) / fx, (v - cy) / fy)))
                T_I, _, _, _ = interp_pose(res.atPose3(X(jj - 1)),
                                           res.atPose3(X(jj)), fac._lam(calib0, k))
                z = float(T_I.compose(calib0.T).transformTo(gtsam.Point3(*P[k]))[2])
                cd.append((float(rc[k]), float(fov), z))
        corner_diag = cd
    sols = [calib_bases[c].retract(res.atVector(_Cc(c))) for c in range(len(cams))]
    marg, sigC, sigI = _marginal_sigmas(graph, res, calib_bases, len(cams), cfg)

    # by what reached the final graph: a frame with no factor left, whichever check removed it, is
    # a no-pose frame; corners rejected from a frame that stayed are outliers
    final = [{(jj, f.dt_frame): f.n for cc, jj, f in cam_factors if cc == c}
             for c in range(len(cams))]
    accounting = _guarded_accounting(len(cams), lambda: _corner_accounting(
        cam_factors, res, sols, image_sizes,
        [_frame_fate(cams[c], tried_ts[c], [frame_time[c][k] for k in final[c]])
         for c in range(len(cams))],
        [sum(n_before_reject[c][k] - n for k, n in final[c].items()) for c in range(len(cams))]))
    cameras = []
    for c in range(len(cams)):
        m = fcam == c
        e = {'calib': sols[c], 'calib_sigma': sigC[c],
             **reproj_entry(corner_px[corner_cam == c], cfg.sigma_px),
             'n_frames': int(m.sum()), 'n_dropped': n_dropped[c],
             'image_size': image_sizes[c],
             'corner_accounting': accounting[c]}
        if c > 0:
            e['T_CtoC0'] = sols[0].T.inverse().compose(sols[c].T)
        _note_kb_domain(e, c)
        cameras.append(e)

    imus = _aux_imu_entries(res, marg, aux_data, aux_factors)

    diagnostics = {
        'gravity': cfg.gravity, 'gravity_source': cfg.gravity_source, 'gravity_site': cfg.gravity_site,
        'gravity_place': cfg.gravity_place,
        'structure': 'feature', 'sfm_mode': sfm_mode, 'iterations': n_iters,
        **reproj_entry(corner_px, cfg.sigma_px),
        'reproj_frame_cam': fcam, 'reproj_per_frame': rms, 'n_states': len(t_states),
        'frame_meta': [(c, f.n, float(f.dt_frame), float(f.dt))
                       for (c, _jj, f) in cam_factors],   # (cam, n_corners, dt_frame, dt)
        'reproj_pre_per_cam': pre_per_cam,      # seed reproj per cam (diagnostic)
        'outlier_drops': drop_log,              # corners dropped per reject round
        'cam0_corner_diag': corner_diag,        # [(residual_px, fov_deg)] when cfg.corner_diag
        'n_landmarks': len(landmarks), 'n_vision_factors': len(cam_factors),
        'imu_intrinsics': cfg.imu_intrinsics, 'seed_report': seed0.report,
    }
    _tga_diagnostics(diagnostics, res, marg)
    return CamImuCalibResult(
        calib=sols[0], calib_sigma=sigC[0],
        **_bias_fields(res, len(t_states)),
        imu_intr_xi=np.asarray(res.atVector(I0), float), imu_intr_sigma=sigI,
        imu_model=imu_model_name(_imu_model(cfg)),
        T_BinW=seed0.T_BinW, seed=seed0, values=res, t_states=t_states,
        diagnostics=diagnostics,
        ok=bool(seed0.report.get('g_norm_ok', False)
                and np.isfinite(diagnostics['reproj_rms_px'])),
        cameras=cameras, imus=imus)


# ---------------------------------------------------------------------------
def _kb_domain_lines(r) -> list:
    """format_report: the valid-domain line of each kb4 camera."""
    out = []
    for c, e in enumerate(r.cameras or []):
        info = e.get('valid_domain')
        if info is not None:
            from mvis.camera.kb_domain import format_kb_valid_domain, kb_domain_tag
            tag = kb_domain_tag(info)
            tag = f"  [{tag}]" if tag else ""
            out.append(f"cam{c} {format_kb_valid_domain(info, e['calib'].model)}{tag}")
    return out


def format_report(r: CamImuCalibResult) -> str:
    """Human-readable calibration report (step 8)."""
    c, sig = r.calib, r.calib_sigma
    rpy = np.rad2deg(log_so3(c.T.rotation().matrix()))
    p = np.asarray(c.T.translation(), float)
    d = r.diagnostics
    names = dist_names(c.model)
    lines = [
        "== joint camera-IMU calibration ==",
        f"states: {d['n_states']}  iters: {d['iterations']}  "
        f"cost {d['err_init']:.1f} -> {d['err_final']:.1f}",
        f"reproj RMS {d['reproj_rms_px']:.3f} px   "
        f"chi2/dof cam {d['cam_chi2_dof']:.2f}  imu {d['imu_chi2_dof']:.2f}",
        "-- camera --",
        f"intr: fx {c.intr[0]:.2f}  fy {c.intr[1]:.2f}  "
        f"cx {c.intr[2]:.2f}  cy {c.intr[3]:.2f}",
        (f"dist ({' '.join(names)}): " if names is not None else "dist: ")
        + np.array2string(c.intr[4:], precision=5),
        f"intr sigma: {np.array2string(sig[8:8 + c.n_intr], precision=4)}",
        *_kb_domain_lines(r),
        f"td: {1e3 * c.td:+.2f} ms (sigma {1e3 * sig[6]:.2f})   "
        f"tr: {1e3 * c.tr:.2f} ms" + (" [pinned]" if sig[7] < 1e-6 else ""),
        f"T_CtoI rot (log, deg): {np.array2string(rpy, precision=3)} "
        f"(sigma {np.array2string(np.rad2deg(sig[0:3]), precision=3)})",
        f"T_CtoI trans (m): {np.array2string(p, precision=4)} "
        f"(sigma {np.array2string(sig[3:6], precision=4)})",
        "-- imu --",
        *(["gravity: " + describe_gravity(d['gravity'], d.get('gravity_source', 'default'), d.get('gravity_site'),
                                          d.get('gravity_place'))]
          if d.get("gravity") is not None else []),
        f"b_g: {np.array2string(r.b_g, precision=5)}  "
        f"drift {np.array2string(r.b_g_drift, precision=5)}",
        f"b_a: {np.array2string(r.b_a, precision=4)}  "
        f"drift {np.array2string(r.b_a_drift, precision=4)}",
    ]
    if r.imu_intr_sigma is None:
        lines.append("Dw/Da/Tg/R_acc: pinned (phase A)")
    else:
        xi, s = r.imu_intr_xi, r.imu_intr_sigma
        sl = intrinsic_block_slices(parse_imu_model(getattr(r, "imu_model", "rpng")))

        def _blk(name, label):
            if name not in sl:
                return None
            b = sl[name]
            pinned = np.max(s[b]) < 1e-6
            tail = " [pinned]" if pinned else f" (sigma {np.array2string(s[b], precision=4)})"
            return f"{label}: {np.array2string(xi[b], precision=4)}{tail}"

        lines += [ln for ln in (_blk("dw", "d_Dw"), _blk("da", "d_Da"),
                                _blk("tg", "d_Tg"), _blk("r", "d_R_acc"))
                   if ln is not None]
    sr = (r.diagnostics or {}).get("seed_report") or {}
    failed = [g for g in ("g_norm_ok", "scale_ok") if sr.get(g) is False]
    if failed:
        lines.append(f"init gate(s) failed (seed only, not in ok): {', '.join(failed)}  "
                     f"|g| {sr.get('g_norm_ls', float('nan')):.3f}  scale {sr.get('scale', float('nan')):.3f}")
    lines.append(f"ok: {r.ok}")
    return "\n".join(lines)


def result_to_dict(r: CamImuCalibResult) -> dict:
    """JSON/yaml-serializable result (step 8 machine output)."""
    c = r.calib
    return {
        'camera': {
            'intrinsics': c.intr.tolist(),
            'is_fisheye': bool(c.is_fisheye), 'model': str(c.model),
            'td_s': float(c.td), 'tr_s': float(c.tr),
            'T_CtoI': {
                'R': c.T.rotation().matrix().tolist(),
                'p': np.asarray(c.T.translation(), float).tolist()},
            'sigma_xi16': r.calib_sigma.tolist(),
        },
        'imu': {
            'b_g': r.b_g.tolist(), 'b_a': r.b_a.tolist(),
            'b_g_drift': r.b_g_drift.tolist(),
            'b_a_drift': r.b_a_drift.tolist(),
            'intrinsics_xi24': r.imu_intr_xi.tolist(),
            'intrinsics_pinned': r.imu_intr_sigma is None,
        },
        'diagnostics': {
            'reproj_rms_px': float(r.diagnostics['reproj_rms_px']),
            'reproj_median_px': (float(r.diagnostics['reproj_median_px'])
                                 if r.diagnostics.get('reproj_median_px') is not None else None),
            'reproj_p95_px': (float(r.diagnostics['reproj_p95_px'])
                              if r.diagnostics.get('reproj_p95_px') is not None else None),
            'cam_chi2_dof': r.diagnostics['cam_chi2_dof'],
            'imu_chi2_dof': r.diagnostics['imu_chi2_dof'],
            'iterations': int(r.diagnostics['iterations']),
            'n_states': int(r.diagnostics['n_states']),
            'n_factors': r.diagnostics.get('n_factors', {}),
            'timings': {k: float(v) for k, v in
                        r.diagnostics.get('timings', {}).items()},
            'seed_td_s': float(r.seed.td),       # base cam-IMU INIT td (I1)
            'ok': bool(r.ok),
            'vision_pose': r.diagnostics.get('vision_pose', 'interp'),
            'vision_imu': r.diagnostics.get('vision_imu'),   # E8 Δ / Dw-corr stats
            'sensor_clock': r.diagnostics.get('sensor_clock'),  # retimed cameras
            'lm_no_step': r.diagnostics.get('lm_no_step'),      # LM stopped at iteration 0
            'lm_rescue': r.diagnostics.get('lm_rescue'),        # LM stopped stuck at its damping cap
        },
        'cameras': [{
            'intrinsics': e['calib'].intr.tolist(),
            'is_fisheye': bool(e['calib'].is_fisheye), 'model': str(e['calib'].model),
            'td_s': float(e['calib'].td), 'tr_s': float(e['calib'].tr),
            'T_CtoI': {
                'R': e['calib'].T.rotation().matrix().tolist(),
                'p': np.asarray(e['calib'].T.translation(), float).tolist()},
            'sigma_xi16': np.asarray(e['calib_sigma'], float).tolist(),
            'reproj_rms_px': float(e['reproj_rms_px']),
            'n_frames': int(e['n_frames']),
            **({'T_CtoC0': {
                    'R': e['T_CtoC0'].rotation().matrix().tolist(),
                    'p': np.asarray(e['T_CtoC0'].translation(),
                                    float).tolist()},
                'seed_gates': e['seed_gates']} if 'T_CtoC0' in e else {}),
        } for e in r.cameras],
        'imus': [{
            'name': m['name'],
            'T_IstoI': {
                'R': m['T_IstoI'].rotation().matrix().tolist(),
                'p': np.asarray(m['T_IstoI'].translation(), float).tolist()},
            'td_s': float(m['td']),
            'b_g_s': np.asarray(m['b_g_s'], float).tolist(),
            'b_a_s': np.asarray(m['b_a_s'], float).tolist(),
            'intr_xi24': np.asarray(m['intr_xi24'], float).tolist(),
            'sigma_xi31': np.asarray(m['calib_sigma'], float).tolist(),
            'chi2_dof': float(m['chi2_dof']),
            'seed_report': {k: (bool(v) if isinstance(v, (bool, np.bool_))
                                else float(v))
                            for k, v in m['seed_report'].items()},
        } for m in r.imus],
    }
