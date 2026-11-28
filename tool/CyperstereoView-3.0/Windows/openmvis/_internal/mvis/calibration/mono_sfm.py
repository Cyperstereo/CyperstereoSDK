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
"""Up-to-scale monocular SfM from anonymous board detections (F0/F1).

Feature-only camera-IMU initialization: the board's 2-D corner detections are
used as generic tracked features (tag IDs give perfect correspondences for
free) while the known 3-D geometry is WITHHELD — the pose track comes out in
an arbitrary structure frame `S` with arbitrary scale, exactly the
feature-only / VINS-Mono situation. The scale-unknown alignment (I3) then
recovers the metric scale from the IMU (‖g‖ anchors the meters).

Because every feature lies on the board, the scene is strictly PLANAR — the
essential matrix is degenerate, so the bootstrap is homography-based by
construction (decompose + cheirality + third-view disambiguation).

Pipeline (`planar_sfm`):
  F0  normalize detections to undistorted normalized coords (intrinsics known)
  F1  parallax-gated keyframes → H-bootstrap pair → triangulate →
      incremental robust PnP + parallax-gated triangulation →
      global gtsam BA (Huber) → outlier pass → re-BA

Output: the BA KEYFRAME pose track `T_CtoS` (camera in structure frame) —
same interface as `pnp_pose_track`'s `T_CtoB`, so the whole I1→I4 init
consumes it unchanged (with `board_mode=False`). Keyframes only, on purpose:
re-PnP-ing every frame against the map re-adds per-frame detection noise on
top of the BA-optimal poses and measurably worsens the alignment's scale.

GEOMETRY SENSITIVITY: a small planar target at large depth is the worst case
for monocular structure (per-pose depth is ill-conditioned → the trajectory
bends; measured: board@1.5 m → ~5 cm pose noise & ~7% scale error at 0.5 px,
board@3 m → 20 cm & ~20%+). Natural 3-D scenes (the real feature-only use
case) are far better conditioned; for board-as-anonymous-features keep the
board close.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import gtsam

from mvis.core.camera_models import legacy_model


# ---------------------------------------------------------------------------
# F0 — anonymous normalized tracks
# ---------------------------------------------------------------------------
def normalize_detections(ids_list, uv_list, cal_vec, is_fisheye: bool = False,
                         model: str | None = None):
    """Undistort pixel detections to normalized image coordinates.

    The corner IDs are kept purely as TRACK ids (correspondence); no board
    geometry enters. Returns a list of (ids (n,), xy (n, 2)) per frame.

    Radtan uses OpenCV undistortPoints. Every other model uses the camera model's
    own inverse, dropping corners it can't unproject (for equidistant, those past
    the polynomial's first branch), with ids filtered to match. `model` defaults
    from `is_fisheye`.
    """
    from mvis.camera.normalized_points import normalized_points

    cal_vec = np.asarray(cal_vec, float)
    if model is None:
        model = legacy_model(is_fisheye, cal_vec)
    out = []
    for ids, uv in zip(ids_list, uv_list):
        uv = np.asarray(uv, np.float64).reshape(-1, 2)
        if len(uv) == 0:
            out.append((np.array([], int), np.zeros((0, 2))))
            continue
        if model == "radtan":           # cv2 takes radtan only
            xy, _ = normalized_points(uv, model, cal_vec, backend="cv2")
            out.append((np.asarray(ids, int), xy))
        else:                           # bearings -> normalized
            xy, valid = normalized_points(uv, model, cal_vec)
            out.append((np.asarray(ids, int)[valid], xy[valid]))
    return out


# ---------------------------------------------------------------------------
# F1 — planar keyframe SfM
# ---------------------------------------------------------------------------
@dataclass
class SfmResult:
    t: np.ndarray                 # (F,) timestamps of frames with poses
    poses: list                   # [gtsam.Pose3 T_CtoS] — up to scale!
    points: dict = field(default_factory=dict)   # track_id -> (3,) in S
    kf_idx: np.ndarray = None     # indices (into t) of the BA keyframes
    reproj_rms: float = 0.0       # BA reprojection RMS (normalized coords)
    planarity_rms: float = 0.0    # plane-fit residual of the map (S units)
    parallax_med: float = 0.0     # median bootstrap parallax (normalized)


def _common(ids_a, xy_a, ids_b, xy_b):
    """Shared track ids between two frames -> matched coordinate arrays."""
    common, ia, ib = np.intersect1d(ids_a, ids_b, return_indices=True)
    return common, xy_a[ia], xy_b[ib]


def _pose_from_rt(R_StoC, t_StoC) -> gtsam.Pose3:
    """cv2 convention [R|t] maps S->camera; return T_CtoS."""
    T = np.eye(4)
    T[:3, :3] = R_StoC
    T[:3, 3] = np.asarray(t_StoC, float).ravel()
    return gtsam.Pose3(np.linalg.inv(T))


def _triangulate(P0, P1, x0, x1):
    """cv2 linear triangulation, normalized coords; returns (N, 3)."""
    import cv2
    X = cv2.triangulatePoints(P0, P1, x0.T.astype(float), x1.T.astype(float))
    return (X[:3] / X[3]).T


def _select_keyframes(tracks, t_cam, parallax_thresh, min_corners):
    """Parallax-gated keyframe indices (always includes usable frame 0)."""
    kf = []
    last = None
    for i, (ids, xy) in enumerate(tracks):
        if len(ids) < min_corners:
            continue
        if last is None:
            kf.append(i)
            last = i
            continue
        common, xa, xb = _common(*tracks[last], ids, xy)
        if len(common) < min_corners:
            kf.append(i)            # track break (visibility gap) — restart
            last = i
            continue
        if np.median(np.linalg.norm(xb - xa, axis=1)) > parallax_thresh:
            kf.append(i)
            last = i
    return kf


def _bootstrap_pair(tracks, kf, parallax_boot, min_corners):
    """First keyframe pair with strong parallax + its homography decomposition.

    Returns (i_kf0, i_kf1, T_CtoS list for both, points dict) or None.
    The structure frame S = camera frame of kf0; scale set by ‖t‖ = 1.
    """
    import cv2

    for a in range(len(kf) - 1):
        ia = kf[a]
        for b in range(a + 1, min(a + 40, len(kf))):
            ib = kf[b]
            common, xa, xb = _common(*tracks[ia], *tracks[ib])
            if len(common) < max(min_corners, 8):
                continue
            med_par = np.median(np.linalg.norm(xb - xa, axis=1))
            if med_par < parallax_boot:
                continue
            H, inl = cv2.findHomography(xa, xb, cv2.RANSAC, 2e-3)
            if H is None or inl.sum() < max(min_corners, 8):
                continue
            # planar scene: decompose H (K = I on normalized coords)
            _n, Rs, ts, _norms = cv2.decomposeHomographyMat(H, np.eye(3))
            cands = []
            for R, t in zip(Rs, ts):
                t = t.ravel()
                if np.linalg.norm(t) < 1e-9:
                    continue        # pure-rotation solution — no scale
                t = t / np.linalg.norm(t)          # structure scale: ‖t‖ = 1
                P0 = np.hstack([np.eye(3), np.zeros((3, 1))])
                P1 = np.hstack([R, t.reshape(3, 1)])
                X = _triangulate(P0, P1, xa, xb)
                z0 = X[:, 2]
                z1 = (X @ R.T + t)[:, 2]
                if np.sum((z0 > 0) & (z1 > 0)) >= 0.8 * len(xa):
                    cands.append((R, t, X))
            if not cands:
                continue
            # THIRD-VIEW disambiguation — the planar two-fold ambiguity passes
            # two-view cheirality on BOTH solutions; only other viewpoints
            # separate the true one from its mirror. Use viewpoints SPREAD
            # across the whole window (near-bootstrap views barely separate
            # the two) and require a decisive margin, else try the next pair.
            if len(cands) > 1:
                pool = [k for k in kf if k not in (ia, ib)]
                step = max(1, len(pool) // 8)
                thirds = pool[::step][:8]
                scores = []
                for R, t, X in cands:
                    pts = {int(tid): X[j] for j, tid in enumerate(common)}
                    errs = []
                    for ic in thirds:
                        got = _pnp_on_map(*tracks[ic], pts, min_corners)
                        if got is not None:
                            errs.append(got[1])
                    scores.append(np.median(errs) if errs else np.inf)
                order_c = np.argsort(scores)
                s_best, s_2nd = scores[order_c[0]], scores[order_c[1]]
                if not np.isfinite(s_best) or s_best > 0.5 * s_2nd:
                    continue                       # ambiguous — next pair
                R, t, X = cands[order_c[0]]
            else:
                R, t, X = cands[0]
            pts = {int(tid): X[j] for j, tid in enumerate(common)}
            poses = {ia: gtsam.Pose3(), ib: _pose_from_rt(R, t)}
            return ia, ib, poses, pts, med_par
    return None


def _bootstrap_pair_essential(tracks, kf, parallax_boot, min_corners):
    """First strong-parallax keyframe pair + its ESSENTIAL-matrix pose.

    For GENERAL (non-planar / 3D) scenes — e.g. ArUco tags spread across several
    boards. Returns (i_kf0, i_kf1, T_CtoS poses, points dict, med_parallax) or
    None. Structure frame S = camera frame of kf0; scale set by ‖t‖ = 1
    (recoverPose returns a unit translation). recoverPose already disambiguates
    the four E-decompositions by cheirality, so no third-view step is needed
    (unlike the planar homography's two-fold ambiguity)."""
    import cv2

    for a in range(len(kf) - 1):
        ia = kf[a]
        for b in range(a + 1, min(a + 40, len(kf))):
            ib = kf[b]
            common, xa, xb = _common(*tracks[ia], *tracks[ib])
            if len(common) < max(min_corners, 8):
                continue
            med_par = np.median(np.linalg.norm(xb - xa, axis=1))
            if med_par < parallax_boot:
                continue
            E, inl = cv2.findEssentialMat(
                xa, xb, focal=1.0, pp=(0.0, 0.0),
                method=cv2.RANSAC, prob=0.999, threshold=2e-3)
            if E is None or E.shape != (3, 3):
                continue                       # no E, or stacked candidates
            if inl is None or int(inl.sum()) < max(min_corners, 8):
                continue
            _n_in, R, t, _mask = cv2.recoverPose(E, xa, xb)
            t = t.ravel()
            if np.linalg.norm(t) < 1e-9:
                continue
            t = t / np.linalg.norm(t)          # structure scale: ‖t‖ = 1
            P0 = np.hstack([np.eye(3), np.zeros((3, 1))])
            P1 = np.hstack([R, t.reshape(3, 1)])
            X = _triangulate(P0, P1, xa, xb)
            z0 = X[:, 2]
            z1 = (X @ R.T + t)[:, 2]
            good = (z0 > 0) & (z1 > 0)         # cheirality on both views
            if int(good.sum()) < max(min_corners, 8):
                continue
            pts = {int(tid): X[j] for j, tid in enumerate(common) if good[j]}
            poses = {ia: gtsam.Pose3(), ib: _pose_from_rt(R, t)}
            return ia, ib, poses, pts, med_par
    return None


def _pnp_on_map(ids, xy, points, min_corners,
                ransac_thresh: float = 4e-3):
    """Robust 3D-2D PnP against the current map (normalized coords, K = I).

    RANSAC + iterative refinement on inliers — under detection noise a plain
    iterative PnP occasionally glitches badly (planar ambiguity), and one bad
    pose in the full-rate track wrecks the downstream alignment.
    Returns (T_CtoS, inlier reproj RMS in normalized units) or None.
    """
    import cv2

    sel = [j for j, tid in enumerate(ids) if tid in points]
    if len(sel) < min_corners:
        return None
    obj = np.array([points[ids[j]] for j in sel], float)
    img = np.asarray(xy, float)[sel]
    try:
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            obj, img, np.eye(3), None, reprojectionError=ransac_thresh,
            iterationsCount=200, flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error:
        return None
    if not ok or inl is None or len(inl) < min_corners:
        return None
    inl = inl.ravel()
    ok, rvec, tvec = cv2.solvePnP(obj[inl], img[inl], np.eye(3), None,
                                  rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    proj = (obj[inl] @ R.T + tvec.ravel())
    proj = proj[:, :2] / proj[:, 2:3]
    rms = float(np.sqrt(np.mean((proj - img[inl]) ** 2)))
    return _pose_from_rt(R, tvec), rms


def planar_sfm(t_cam, ids_list, xy_list, *, parallax_kf: float = 0.03,
               parallax_boot: float = 0.15, min_corners: int = 8,
               huber_px: float = 2.0, focal_px: float = 450.0) -> SfmResult:
    """Up-to-scale keyframe SfM on (anonymous, planar) normalized tracks.

    Args:
        t_cam, ids_list, xy_list: anonymous normalized tracks (from
            `normalize_detections`).
        parallax_kf / parallax_boot: keyframe / bootstrap median-parallax
            thresholds (normalized units).
        huber_px, focal_px: Huber threshold expressed in pixels at a nominal
            focal length (converted to normalized units).

    Returns SfmResult with the BA KEYFRAME `T_CtoS` poses — arbitrary scale,
    structure frame = first keyframe.
    """
    t_cam = np.asarray(t_cam, float)
    tracks = list(zip(ids_list, xy_list))

    kf = _select_keyframes(tracks, t_cam, parallax_kf, min_corners)
    if len(kf) < 3:
        raise ValueError(f"only {len(kf)} keyframes — not enough for SfM")

    boot = _bootstrap_pair(tracks, kf, parallax_boot, min_corners)
    if boot is None:
        raise ValueError("bootstrap failed — insufficient parallax for a "
                         "homography baseline")
    i0, i1, kf_poses, points, med_par = boot
    return _sfm_core(t_cam, tracks, kf, i0, i1, kf_poses, points, med_par,
                     parallax_boot=parallax_boot, min_corners=min_corners,
                     huber_px=huber_px, focal_px=focal_px)


def _sfm_core(t_cam, tracks, kf, i0, i1, kf_poses, points, med_par, *,
              parallax_boot, min_corners, huber_px, focal_px) -> SfmResult:
    """Shared incremental SfM + global BA after a two-view bootstrap.

    Structure-agnostic: identical for the planar (homography) and general
    (essential-matrix) bootstraps — only `_bootstrap_pair*` differs."""

    # --- incremental: pose each keyframe by PnP, triangulate new tracks ---
    order = [i for i in kf if i not in (i0, i1)]
    # process keyframes sorted by |index - bootstrap| so the map grows outward
    order.sort(key=lambda i: min(abs(i - i0), abs(i - i1)))
    for i in order:
        ids, xy = tracks[i]
        got = _pnp_on_map(ids, xy, points, min_corners)
        if got is None:
            continue
        pose, _rms = got
        kf_poses[i] = pose
        # triangulate tracks not yet in the map — ONLY against a mapped
        # keyframe with sufficient parallax for those points (low-baseline
        # triangulation under noise produces poison points)
        new = [tid for tid in ids if tid not in points]
        if new:
            for j in sorted(kf_poses, key=lambda j: -abs(j - i)):
                common, xa, xb = _common(*tracks[j], ids, xy)
                mask = np.isin(common, new)
                if mask.sum() < 1:
                    continue
                par = np.linalg.norm(xb[mask] - xa[mask], axis=1)
                good = mask.copy()
                good[mask] = par > parallax_boot
                if good.sum() < 1:
                    continue
                Ta = np.linalg.inv(kf_poses[j].matrix())
                Tb = np.linalg.inv(pose.matrix())
                X = _triangulate(Ta[:3, :], Tb[:3, :], xa[good], xb[good])
                for tid, Xj in zip(np.asarray(common)[good], X):
                    points[int(tid)] = Xj
                break

    # --- global BA (gtsam; normalized coords => Cal3_S2 = identity) ---
    X = gtsam.symbol_shorthand.X
    L = gtsam.symbol_shorthand.L
    cal = gtsam.Cal3_S2(1.0, 1.0, 0.0, 0.0, 0.0)
    sigma_n = huber_px / focal_px
    base = gtsam.noiseModel.Isotropic.Sigma(2, sigma_n)
    robust = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345), base)

    kf_ids = sorted(kf_poses)
    key_of = {i: k for k, i in enumerate(kf_ids)}

    def _run_ba(obs):
        """One global BA over the given observations [(kf_i, tid, xy)]."""
        graph = gtsam.NonlinearFactorGraph()
        vals = gtsam.Values()
        used = set()
        for i in kf_ids:
            vals.insert(X(key_of[i]), kf_poses[i])
        for i, tid, x in obs:
            graph.add(gtsam.GenericProjectionFactorCal3_S2(
                gtsam.Point2(*x), robust, X(key_of[i]), L(tid), cal))
            used.add(tid)
        for tid in used:
            vals.insert(L(tid), gtsam.Point3(*points[tid]))
        # gauge: fix pose0 (6 DOF) + the bootstrap baseline length (scale DOF)
        graph.add(gtsam.PriorFactorPose3(
            X(key_of[i0]), kf_poses[i0],
            gtsam.noiseModel.Isotropic.Sigma(6, 1e-9)))
        rng = float(np.linalg.norm(
            np.asarray(kf_poses[i1].translation(), float)
            - np.asarray(kf_poses[i0].translation(), float)))
        graph.add(gtsam.RangeFactorPose3(
            X(key_of[i0]), X(key_of[i1]), rng,
            gtsam.noiseModel.Isotropic.Sigma(1, 1e-9)))
        res = gtsam.LevenbergMarquardtOptimizer(graph, vals).optimize()
        for i in kf_ids:
            kf_poses[i] = res.atPose3(X(key_of[i]))
        for tid in used:
            points[tid] = np.asarray(res.atPoint3(L(tid)), float)
        return res, used

    obs = []
    for i in kf_ids:
        ids, xy = tracks[i]
        for tid, x in zip(ids, xy):
            if int(tid) in points:
                obs.append((i, int(tid), x))

    _res, used_pts = _run_ba(obs)

    def _resid(i, tid, x):
        T = np.linalg.inv(kf_poses[i].matrix())
        pc = T[:3, :3] @ points[tid] + T[:3, 3]
        if pc[2] <= 1e-6:
            return np.inf
        return float(np.linalg.norm(pc[:2] / pc[2] - x))

    # outlier pass: drop observations beyond 3σ and re-BA — Huber bounds their
    # influence but heavy-tailed leftovers still bend the planar map
    r = np.array([_resid(i, tid, x) for i, tid, x in obs])
    keep = r < 3.0 * max(np.median(r) * 1.4826, sigma_n)
    if keep.sum() < len(obs):
        obs = [o for o, k in zip(obs, keep) if k]
        _res, used_pts = _run_ba(obs)
        r = np.array([_resid(i, tid, x) for i, tid, x in obs])

    reproj_rms = float(np.sqrt(np.mean(r ** 2)) * focal_px)

    # --- output: the BA KEYFRAME track only. Re-PnP-ing every frame against
    # the map would re-add per-frame detection noise on top of the BA-optimal
    # poses (measured: it doubles the alignment's scale attenuation); the
    # parallax-gated keyframes are dense enough (~5-10 Hz) for the init's
    # td/rotation/alignment stages, whose interval-mean pairing is
    # rate-agnostic. Velocity-jump gate (percentile-only — structure units
    # are arbitrary) as cheap insurance.
    t_out = np.array([t_cam[i] for i in kf_ids])
    poses_out = [kf_poses[i] for i in kf_ids]
    if len(poses_out) > 10:
        p = np.array([np.asarray(P.translation(), float) for P in poses_out])
        v = np.linalg.norm(np.diff(p, axis=0), axis=1) / np.maximum(
            np.diff(t_out), 1e-9)
        thr = 3.0 * np.percentile(v, 95)
        from mvis.calibration.pose_alignment import velocity_jump_inliers
        ok = velocity_jump_inliers(v, thr)
        t_out = t_out[ok]
        poses_out = [P for P, o in zip(poses_out, ok) if o]

    # planarity of the final map (free quality gate — geometry was withheld,
    # so coplanarity is evidence, not construction)
    P = np.array([points[tid] for tid in sorted(used_pts)])
    Pc = P - P.mean(0)
    planarity = float(np.linalg.svd(Pc, compute_uv=False)[-1]
                      / max(np.sqrt(len(P)), 1))

    return SfmResult(
        t=t_out, poses=poses_out, points=points,
        kf_idx=np.array(kf_ids, int), reproj_rms=reproj_rms,
        planarity_rms=planarity, parallax_med=float(med_par))


def general_sfm(t_cam, ids_list, xy_list, *, parallax_kf: float = 0.03,
                parallax_boot: float = 0.08, min_corners: int = 8,
                huber_px: float = 2.0, focal_px: float = 450.0) -> SfmResult:
    """Up-to-scale keyframe SfM for a GENERAL (non-planar, 3D) scene.

    Identical to `planar_sfm` except the two-view bootstrap uses the essential
    matrix (`findEssentialMat` + `recoverPose`), which is well-conditioned when
    the features are NOT coplanar (and degenerate when they ARE — use
    `planar_sfm` then). Intended for ArUco tags spread across several boards in
    3D (data association from globally-unique tag IDs). All downstream stages
    (incremental PnP, triangulation, global BA, outlier pass) are shared with
    `planar_sfm` via `_sfm_core`.
    """
    t_cam = np.asarray(t_cam, float)
    tracks = list(zip(ids_list, xy_list))

    kf = _select_keyframes(tracks, t_cam, parallax_kf, min_corners)
    if len(kf) < 3:
        raise ValueError(f"only {len(kf)} keyframes — not enough for SfM")

    boot = _bootstrap_pair_essential(tracks, kf, parallax_boot, min_corners)
    if boot is None:
        raise ValueError("bootstrap failed — insufficient parallax for an "
                         "essential-matrix baseline")
    i0, i1, kf_poses, points, med_par = boot
    return _sfm_core(t_cam, tracks, kf, i0, i1, kf_poses, points, med_par,
                     parallax_boot=parallax_boot, min_corners=min_corners,
                     huber_px=huber_px, focal_px=focal_px)
