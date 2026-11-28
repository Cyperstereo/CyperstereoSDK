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
"""Per-pose camera images for the moving frustum in the cam-only 3D plot.

Each board pose gets a small grayscale image for the frustum's image plane:
the real detection thumbnail when `thumbnails/cam<N>/index.csv` maps the pose
time to one (written by `DetectionPipeline._write_thumbnail`), otherwise a
rendered view — the board projected through the calibrated model at that pose
with the detected corners on top — so older runs and regenerated reports still
show what the camera saw.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import gtsam
import numpy as np

from mvis.calibration.factors.camera_projection import k_cal_vec, project_observation
from mvis.io.detection_pipeline import THUMB_INDEX_NAME

logger = logging.getLogger(__name__)

# Pose time <-> thumbnail time match (thumbnail stamps are the loader's frame
# stamps, so a real match is exact up to the 1e-9 s index precision).
_MATCH_TOL_S = 1e-3

# Rendered-view gray levels: background, board area, board points, detections
_BG, _BOARD, _POINT, _DET = 40, 150, 90, 255


def load_thumb_index(thumb_dir, cam_id):
    """(timestamps, jpg paths) from `cam<N>/index.csv`, or None if absent."""
    if thumb_dir is None:
        return None
    path = Path(thumb_dir) / f"cam{cam_id}" / THUMB_INDEX_NAME
    if not path.exists():
        return None
    ts, paths = [], []
    for line in path.read_text().splitlines():
        try:
            idx, t = line.split(",")
            ts.append(float(t))
            paths.append(path.parent / f"{int(idx):05d}.jpg")
        except ValueError:
            continue
    return (np.array(ts), paths) if ts else None


def _real_image(t, index, size, color=False):
    ts, paths = index
    i = int(np.argmin(np.abs(ts - t)))
    if abs(ts[i] - t) > _MATCH_TOL_S:
        return None
    img = cv2.imread(str(paths[i]), cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
    return None if img is None else cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def _rendered_image(T_board_cam, cam, board, frame, size):
    """Board projected at this pose (calibrated model) + detected corners."""
    w, h = cam.image_size
    s = size[0] / float(w)
    cal_vec = k_cal_vec(cam.distortion_model, cam.intrinsics, cam.distortion)
    pose = gtsam.Pose3(T_board_cam)
    img = np.full((size[1], size[0]), _BG, np.uint8)
    px = []
    for p in board.all_points_3d():
        uv = project_observation(board_point=p, bp0=pose, bp1=pose,
                                 time_meas=0.0, time_0=0.0, time_1=1.0,
                                 cal_vec=cal_vec,
                                 distortion_model=cam.distortion_model)
        if uv is not None and np.all(np.isfinite(uv)):
            px.append(np.asarray(uv, float) * s)
    if len(px) >= 3:
        px = np.array(px)
        hull = cv2.convexHull(np.round(px).astype(np.int32))
        cv2.fillConvexPoly(img, hull, _BOARD)
        for u, v in np.round(px).astype(int):
            if 0 <= u < size[0] and 0 <= v < size[1]:
                img[v, u] = _POINT
    if frame is not None:
        r = max(1, round(size[0] / 64))
        for o in frame.observations:
            u, v = np.round(np.asarray(o.pixel, float) * s).astype(int)
            cv2.circle(img, (int(u), int(v)), r, _DET, -1)
    return img


def pose_images(poses, times, cam, board, frames, thumb_dir=None, width=64):
    """One grayscale image per pose, shape (N, h, width), plus its source.

    `poses` are T_board_cam (4x4) of camera `cam` at `times`; `frames` are that
    camera's FrameObservations (detections drawn on rendered views). Returns
    (images uint8, ["real" | "rendered", ...]).
    """
    w, h = cam.image_size
    size = (int(width), max(2, round(width * h / float(w))))
    index = load_thumb_index(thumb_dir, cam.camera_id)
    by_t = {round(f.timestamp, 6): f for f in frames}
    imgs, src = [], []
    for T, t in zip(poses, times):
        img = _real_image(t, index, size) if index is not None else None
        if img is not None:
            src.append("real")
        else:
            img = _rendered_image(T, cam, board, by_t.get(round(t, 6)), size)
            src.append("rendered")
        imgs.append(img)
    n_real = src.count("real")
    if index is not None and n_real < len(src):
        logger.info(f"frustum images: {n_real}/{len(src)} real thumbnails, "
                    "rest rendered")
    return np.stack(imgs) if imgs else np.zeros((0, size[1], size[0]), np.uint8), src


# Image-coverage grid for the side panel (cells with >= 1 detected corner)
_COV_GRID = (8, 6)


def pose_panel(poses, times, cam, board, frames, thumb_dir=None,
               width=200, max_images=240, quality=70, frame_groups=None):
    """Side-panel data for the cam-only 3D plot, one entry per pose.

    `jpg`: colour JPEGs (base64) of the camera view — real annotated thumbnail,
    else the rendered view with detections in green — for at most `max_images`
    evenly spaced poses (`jpg_idx` = their pose indices). `uv`: detected corner
    pixels per pose (flat ints, image px). `cov`: cumulative image coverage (%
    of an 8x6 grid with a corner seen up to that pose). `n`: cumulative corners.
    `frame_groups` (one list of frames per pose) replaces the per-time frame
    lookup for uv/coverage — a thinned timeline step then still counts every
    frame since the previous step.
    """
    import base64
    w, h = cam.image_size
    size = (int(width), max(2, round(width * h / float(w))))
    index = load_thumb_index(thumb_dir, cam.camera_id)
    by_t = {round(f.timestamp, 6): f for f in frames}
    n_pose = len(poses)
    pick = sorted(set(np.linspace(0, n_pose - 1, min(n_pose, max_images))
                      .round().astype(int).tolist())) if n_pose else []
    jpg = []
    for k in pick:
        img = None
        if index is not None:
            img = _real_image(times[k], index, size, color=True)
        if img is None:
            frame = by_t.get(round(times[k], 6))
            img = cv2.cvtColor(_rendered_image(poses[k], cam, board, None, size),
                               cv2.COLOR_GRAY2BGR)
            if frame is not None:
                s = size[0] / float(w)
                r = max(1, round(size[0] / 80))
                for o in frame.observations:
                    u, v = np.round(np.asarray(o.pixel, float) * s).astype(int)
                    cv2.circle(img, (int(u), int(v)), r, (0, 220, 0), -1)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        jpg.append(base64.b64encode(buf).decode("ascii") if ok else "")
    gx, gy = _COV_GRID
    seen = np.zeros((gy, gx), bool)
    uv, cov, n_cum, n = [], [], [], 0
    for k, t in enumerate(times):
        group = (frame_groups[k] if frame_groups is not None
                 else [by_t.get(round(t, 6))])
        obs = [o for f in group if f is not None for o in f.observations]
        px = (np.round([o.pixel for o in obs]).astype(int)
              if obs else np.zeros((0, 2), int))
        uv.append(px.reshape(-1).tolist())
        if len(px):
            cx = np.clip(px[:, 0] * gx // max(w, 1), 0, gx - 1)
            cy = np.clip(px[:, 1] * gy // max(h, 1), 0, gy - 1)
            seen[cy, cx] = True
        n += len(px)
        cov.append(round(100.0 * seen.mean(), 1))
        n_cum.append(n)
    return dict(w=int(w), h=int(h), jpg=jpg, jpg_idx=pick, uv=uv, cov=cov,
                n=n_cum, grid=list(_COV_GRID))


def match_frames(times, frames, tol):
    """Per pose time: the nearest of `frames` within `tol` s (else None) and the
    time to use for it (the frame's own stamp, so thumbnails match exactly)."""
    ts = np.array([f.timestamp for f in frames])
    out_t, out_f = [], []
    for t in times:
        if len(ts):
            j = int(np.argmin(np.abs(ts - t)))
            if abs(ts[j] - t) <= tol:
                out_t.append(float(ts[j])); out_f.append(frames[j])
                continue
        out_t.append(float(t)); out_f.append(None)
    return out_t, out_f


def camera_views(poses, times, cameras, ext_by_id, ref_id, board, all_frames,
                 thumb_dir=None, time_offsets=None):
    """One view dict per camera (ref first) for plot_3d.plot_camera_poses.

    `poses`/`times` are the ref camera's T_board_cam and stamps; camera c's pose
    is T_board_ref . inv(T_ref<-c extrinsic `ext_by_id[c]`). Each pose is matched
    to camera c's nearest detected frame (within half its median frame period,
    shifted by its time offset). Returns dicts with camera_id, images, panel
    and ids (detected board point ids per pose).
    """
    order = sorted(cameras, key=lambda c: c.camera_id != ref_id)
    max_img = 240 if len(order) == 1 else 160
    views = []
    for cam in order:
        cid = cam.camera_id
        frames = sorted(all_frames.get(cid, []), key=lambda f: f.timestamp)
        if cid == ref_id:
            cposes = list(poses)
        elif cid in ext_by_id:
            inv = np.linalg.inv(np.asarray(ext_by_id[cid], float))
            cposes = [np.asarray(T, float) @ inv for T in poses]
        else:
            continue
        ft = np.array([f.timestamp for f in frames])
        tol = 0.5 * float(np.median(np.diff(ft))) if len(ft) > 1 else 1e-3
        off = float((time_offsets or {}).get(cid, 0.0)) if cid != ref_id else 0.0
        ctimes, cframes = match_frames(np.asarray(times, float) - off, frames, tol)
        imgs, _ = pose_images(cposes, ctimes, cam, board, frames, thumb_dir=thumb_dir)
        panel = pose_panel(cposes, ctimes, cam, board, frames, thumb_dir=thumb_dir,
                           max_images=max_img)
        ids = [[int(o.point_id) for o in f.observations] if f is not None else []
               for f in cframes]
        views.append(dict(camera_id=cid, images=imgs, panel=panel, ids=ids))
    return views
