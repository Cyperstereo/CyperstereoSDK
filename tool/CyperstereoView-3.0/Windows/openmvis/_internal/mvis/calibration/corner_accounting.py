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
"""Per-camera count of what became of each corner, for all models. A diagnostic: it reads results
and feeds no metric or cost.

A corner whose projection fails drops out of the reprojection statistics, and in cam-IMU also out of
the cost (zero rows), so crossing the fold can silently lower a fit's error. Every corner the solve
was given lands in exactly one category:

    used            it had a factor in the final solve and projects at the final fit
    past_fold       it fails to project at the final fit, at or past a KB model's fold
    other_invalid   it fails to project at the final fit for any other reason, such as z <= 0 under
                    equidistant
    outliers        the outlier filter removed it and it still projects at the final fit
    no_pose_corners its frame reached no factor (no pose, too few points, outside the pose timeline),
                    or its pose was discarded after the solve; it was not judged
    not_in_solve    it never reached the final solve: thinned by rate or `corner_subsample`, or
                    dropped by a gate. The remainder, so it must not be negative

    detected = used + past_fold + other_invalid + outliers + no_pose_corners + not_in_solve

Beside those, not added to the sum: `n_corners` is used + past_fold + other_invalid; `filter_removed`
is what the outlier filter removed, so it matches `OutlierFilterStats` before minus after, and
`removed_invalid` is those that fail to project at the final fit; `discarded_pose_corners` is the part
of no_pose_corners that had a factor until final cleanup discarded the pose; `past_fold_pass1` is the
count of past_fold at pass 1's fit, for a fold the outlier rerun moved past them, and raises the alarm
too. A corner removed from a frame with no final pose is judged at its pass 1 pose, with the final fit.

`count_corners` is the one walk over camera-only corners. Cam-IMU walks its factors and classifies
through the same `CornerTally`.
"""

from __future__ import annotations

import logging
import math

import gtsam
import numpy as np

from mvis.core.camera_models import KB_THETA_MODELS, n_dist

logger = logging.getLogger(__name__)


def guarded(count, *args, **kwargs):
    """`count(*args, **kwargs)`, or None if it raises. The count must never decide a calibration: it
    runs inside solves, and a ValueError from one reads as "Stage 5 rerun failed" and reverts it."""
    try:
        return count(*args, **kwargs)
    except Exception:
        logger.exception("corner_accounting failed: this result has no corner count and no "
                         "past-fold alarm. The calibration itself is not affected")
        return None


def warn_past_90(a: dict | None, label: str, log: logging.Logger = logger) -> None:
    """Equidistant has no pixel past 90 deg: say so when corners fell out of its fit for that, and
    name the model that can use them. `a`: one `corner_accounting` dict, None if the count failed."""
    n = (a or {}).get("other_invalid", 0)
    if a and a.get("model") == "equidistant" and n:
        log.warning(f"{label}: {n} corners lie at or past 90 deg, where equidistant has no "
                    f"projection, so they are not in its error. Switch to kb4 to use them")


def not_in_solve(a: dict) -> int:
    """The remainder of `detected` for one `corner_accounting` dict. Negative means the count is
    wrong, so say so."""
    n = int(a["detected"] - a["n_corners"] - a["no_pose_corners"] - a["outliers"])
    if n < 0:
        logger.warning(f"corner_accounting: camera {a.get('camera')}: not_in_solve is {n}, more "
                       f"corners counted than the {a['detected']} detected, so this count is wrong")
    return n


class CornerTally:
    """The category rules, one camera."""

    def __init__(self, camera, model: str, cal_vec):
        self.camera, self.model = camera, model
        self.cal_vec = [float(x) for x in cal_vec]
        self.fold = None
        if model in KB_THETA_MODELS:
            from mvis.calibration.factors.custom_projections import _kb_fold
            self.fold = _kb_fold(tuple(self.cal_vec[4:]))
        self.used = self.past_fold = self.other_invalid = self.outliers = 0
        self.no_pose_frames = self.no_pose_corners = self.discarded_pose_corners = 0
        self.filter_removed = self.removed_invalid = self.past_fold_pass1 = 0
        self.max_theta = self.max_theta_used = None

    def _theta(self, p_cam):
        theta = math.atan2(math.hypot(float(p_cam[0]), float(p_cam[1])), float(p_cam[2]))
        self.max_theta = theta if self.max_theta is None else max(self.max_theta, theta)
        return theta

    def add(self, p_cam, valid: bool) -> None:
        """A corner that had a factor."""
        theta = self._theta(p_cam)
        if valid:
            self.used += 1
            self.max_theta_used = (theta if self.max_theta_used is None
                                   else max(self.max_theta_used, theta))
        elif self.fold is not None and self.fold <= theta < math.pi:
            self.past_fold += 1
        else:
            self.other_invalid += 1

    def add_filter_removed(self, p_cam, valid: bool) -> None:
        """A corner the outlier filter removed; `p_cam` None when no pose exists to judge it at."""
        self.filter_removed += 1
        if p_cam is None:
            self.no_pose_corners += 1
        elif valid:
            self._theta(p_cam)
            self.outliers += 1
        else:
            self.removed_invalid += 1
            self.add(p_cam, False)

    def to_dict(self, image_size=None) -> dict:
        def deg(x):
            return None if x is None else float(math.degrees(x))
        fold_deg, level = None, "none"
        if self.fold is not None:
            from mvis.camera.kb_domain import fold_level, kb_valid_domain
            info = kb_valid_domain(self.cal_vec, image_size)
            level = fold_level(info, self.past_fold + self.past_fold_pass1)
            if info["fold_theta_rad"] is not None:
                fold_deg = deg(info["fold_theta_rad"])
        return {
            "camera": self.camera, "model": self.model,
            "n_corners": self.used + self.past_fold + self.other_invalid,
            "used": self.used, "past_fold": self.past_fold,
            "past_fold_pass1": self.past_fold_pass1,
            "other_invalid": self.other_invalid,
            "no_pose_frames": self.no_pose_frames,
            "no_pose_corners": self.no_pose_corners,
            "discarded_pose_corners": self.discarded_pose_corners,
            "outliers": self.outliers,
            "filter_removed": self.filter_removed,
            "removed_invalid": self.removed_invalid,
            "max_theta_deg": deg(self.max_theta),
            "max_theta_used_deg": deg(self.max_theta_used),
            "fold_deg": fold_deg, "fold_level": level,
        }


class _PoseTrack:
    """The poses of one result, and the one rule for whether a frame has a pose there: a bracket
    with both ends valid. It does not interpolate across a discarded pose."""

    def __init__(self, result, board, image_sizes, bad_poses=None):
        self.result, self.board, self.image_sizes = result, board, image_sizes
        n = len(result.board_pose_times)
        self.order = sorted(range(n), key=lambda i: result.board_pose_times[i])
        self.times = np.array([result.board_pose_times[i] for i in self.order], dtype=np.float64)
        bad = bad_poses or [False] * n
        self.poses = {i: gtsam.Pose3(p) for i, p in enumerate(result.board_poses[:n])
                      if p is not None and not bad[i] and not np.allclose(p, np.eye(4))}
        from mvis.calibration.result_projector import ResultProjector
        self.projector = ResultProjector(result, board, image_sizes)

    def bracket(self, t_frame):
        """(t, i0, i1, t0, t1) for a frame stamp, or None if it has no pose here."""
        from mvis.calibration.factors.camera_projection import find_bracketing_poses, synced_time
        if len(self.order) < 2:
            return None
        t = synced_time(t_frame, self.times, getattr(self.result, "snap_sync_tolerance_s", 0.0))
        b = find_bracketing_poses(t, self.times, self.order)
        if b is None:
            return None
        i0, i1, t0, t1 = b
        if i1 not in self.poses and abs(t - t0) < 1e-9:      # exactly at the valid end
            i1, t1 = i0, t0
        elif i0 not in self.poses and abs(t - t1) < 1e-9:
            i0, t0 = i1, t1
        return (t, i0, i1, t0, t1) if i0 in self.poses and i1 in self.poses else None

    def corner(self, cid, b, obs, cal_vec, model):
        """(camera-frame point, whether it projects with `cal_vec`) at bracket `b`."""
        from mvis.calibration.factors.camera_projection import project_at_pose
        t, i0, i1, t0, t1 = b
        proj = self.projector
        pt = proj.board_point(obs.point_id)
        T = proj.observation_pose(proj.camera(cid), obs, (t, self.poses[i0], self.poses[i1], t0, t1))
        p = np.asarray(T.transformTo(gtsam.Point3(float(pt[0]), float(pt[1]), float(pt[2]))),
                       float).ravel()[:3]
        return p, project_at_pose(T, pt, cal_vec, model) is not None


def count_corners(result, frames, board, image_sizes, *, before_filter=None, after_filter=None,
                  pass1=None, pass1_frames=None, pass1_bad_poses=None, bad_poses=None) -> dict:
    """{camera_id: `corner_accounting`} of `result`. `frames`: what its solve was given.

    After an outlier rerun also give `before_filter` and `after_filter`, the frames each side of
    `filter_outliers` (gates may drop more between that and `frames`: those are `not_in_solve`), and
    `pass1`, `pass1_frames` and `pass1_bad_poses`, the result before the rerun, what its solve was
    given, and the poses final cleanup would discard from it. `bad_poses` treats those poses of
    `result` as discarded."""
    from mvis.calibration.factors.camera_projection import k_cal_vec
    track = _PoseTrack(result, board, image_sizes, bad_poses)
    first = None if pass1 is None else _PoseTrack(pass1, board, image_sizes, pass1_bad_poses)
    factored = result.factor_frames or {}
    out = {}
    for cam in result.cameras:
        cid, model = cam.camera_id, cam.distortion_model
        cal_vec = k_cal_vec(model, cam.intrinsics, cam.distortion)
        tally = CornerTally(cid, model, cal_vec[:4 + n_dist(model)])
        in_solve = {f.timestamp for f in frames.get(cid, [])}
        passed = None if after_filter is None else {
            f.timestamp: {o.point_id for o in f.observations} for f in after_filter.get(cid, [])}
        for frame in (frames if before_filter is None else before_filter).get(cid, []):
            ts = frame.timestamp
            had_factor = float(ts) in factored.get(cid, ())
            b = track.bracket(ts) if had_factor else None
            b_first = first.bracket(ts) if first is not None and b is None else None
            kept = None if passed is None else passed.get(ts, ())
            lost = 0
            for obs in frame.observations:
                if kept is not None and obs.point_id not in kept:
                    src, at = (track, b) if b is not None else (first, b_first)
                    if at is None:
                        tally.add_filter_removed(None, False)
                        lost += 1
                    else:
                        tally.add_filter_removed(*src.corner(cid, at, obs, cal_vec, model))
                elif b is not None:
                    tally.add(*track.corner(cid, b, obs, cal_vec, model))
                elif ts in in_solve:
                    tally.no_pose_corners += 1
                    tally.discarded_pose_corners += had_factor
                    lost += 1
            tally.no_pose_frames += lost > 0
        out[cid] = tally
    if pass1 is not None and any(c.distortion_model in KB_THETA_MODELS for c in result.cameras):
        counted = count_corners(pass1, pass1_frames, board, image_sizes, bad_poses=pass1_bad_poses)
        for cid, tally in out.items():
            tally.past_fold_pass1 = counted.get(cid, {}).get("past_fold", 0)
    return {cid: t.to_dict(image_sizes.get(cid)) for cid, t in out.items()}


def accounting_text(a: dict) -> str:
    """One line for the text reports."""
    th = a.get("max_theta_deg")
    first = a.get("past_fold_pass1", 0)
    s = (f"corners: {a['used']} used, {a['past_fold']} past fold"
         f"{f' ({first} at the first pass)' if first else ''}, "
         f"{a['other_invalid']} other invalid; {a['no_pose_frames']} no-pose frames, "
         f"{a['outliers']} outliers; max theta "
         f"{'n/a' if th is None else f'{th:.1f}'} deg")
    if a.get("fold_deg") is not None:
        s += f", fold {a['fold_deg']:.1f} deg"
    if a.get("fold_level") == "alarm":
        s += "  [ALARM: CORNERS PAST THE FOLD]"
    if a.get("model") == "equidistant" and a["other_invalid"]:
        s += "  [corners past 90 deg have no projection here: switch to kb4]"
    return s
