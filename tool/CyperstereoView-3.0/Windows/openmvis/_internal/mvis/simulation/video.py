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
"""Render an MP4/GIF of a cam-IMU calibration scene — the rig flying its
trajectory, the board(s) fixed in the world, a moving camera frustum, and a
synchronized image panel.

Two modes (added in later tasks): `render_sim_video` (from a SimConfig/scenario)
and `render_calib_video` (from a real calibration result, playing the per-frame
detections + reprojection). This module owns only the animation plumbing; the
static per-frame scene reuses `visualization.py`'s helpers.

Headless (Agg). MP4 via ffmpeg when available, else a GIF fallback (Pillow).
"""
from __future__ import annotations

import os

import numpy as np
import gtsam

from mvis.core.camera_models import legacy_model
from mvis.simulation.visualization import (
    _board_face, _draw_frustum, _equal_aspect,
)

# distinct colors per board (multi-board aware)
_BOARD_COLORS = ["tab:orange", "tab:green", "tab:purple", "tab:cyan",
                 "tab:olive", "tab:pink", "tab:brown", "tab:red"]
# per-camera colors (cam0/base is red to match its bold frustum)
_CAM_COLORS = ["red", "tab:blue", "tab:green", "tab:purple", "tab:orange", "tab:cyan"]


def _draw_optical_axis(ax, pos, R, *, length=0.2, color="tab:blue", lw=1.6):
    """Arrow along the camera optical axis (+z_cam) in world — represents a camera."""
    z = R[:, 2]
    ax.quiver(pos[0], pos[1], pos[2], z[0], z[1], z[2], length=length,
              color=color, linewidth=lw, arrow_length_ratio=0.35)


def _draw_path(ax, cam_path, upto=None):
    """Draw the camera path on a 3-D axis. With ``upto`` (the current frame index) the path
    GROWS: a faint full-path ghost gives spatial context, the traversed ``cam_path[:upto+1]`` is
    drawn bright on top, and a current-position marker sits at ``cam_path[upto]``. ``upto=None``
    draws the whole path bright (static plots). Always marks start (green ●) and end (red ■)."""
    cam_path = np.asarray(cam_path, float)
    n = len(cam_path)
    end = n if upto is None else min(int(upto) + 1, n)
    if upto is not None:                                  # faint full path for context
        ax.plot(cam_path[:, 0], cam_path[:, 1], cam_path[:, 2], "-",
                color="tab:blue", lw=0.8, alpha=0.12)
    ax.plot(cam_path[:end, 0], cam_path[:end, 1], cam_path[:end, 2], "-",
            color="tab:blue", lw=1.4)
    ax.scatter(*cam_path[0], color="tab:green", s=45, marker="o",
               depthshade=False, label="start")
    ax.scatter(*cam_path[-1], color="tab:red", s=45, marker="s",
               depthshade=False, label="end")
    if upto is not None and 0 <= int(upto) < n:           # current position
        ax.scatter(*cam_path[int(upto)], color="black", s=32, marker="^",
                   depthshade=False, label="now")


def camera_source_label(cam_topics, src_path, src_format, c):
    """Short per-camera source id for a video panel: the ROS topic (ros1) or the euroc cam image
    folder, falling back to ``cam<c>``. Used to show WHERE each panel's images came from."""
    if cam_topics and c < len(cam_topics):
        return cam_topics[c]
    if src_format == "euroc" and src_path:
        return f"{os.path.basename(src_path.rstrip('/'))}/cam{c}"
    return f"cam{c}"


def world_from_capture(cap, stride):
    """JSON-ready world-result dict from a `twin.capture_solve` capture: the IMU world states
    (T_ItoW), the board pose and points, EVERY camera (calib, td, readout and its detections at
    every `stride`-th frame) and the aux IMU extrinsics: the input of a real-run world video,
    and small enough to cache as JSON so a re-render needs no re-solve."""
    from mvis.calibration.pipeline_cam_imu import X
    r, cams = cap['r'], cap['cams']
    pts_B = np.asarray(cap['board_points'], float)
    ts = np.asarray(r.t_states, float)
    T_states = [r.values.atPose3(X(i)).matrix().tolist() for i in range(len(ts))]
    cameras = []
    for c, e in enumerate(r.cameras):
        cal, cam = e["calib"], cams[c]
        keep = list(range(0, len(cam['t_cam']), max(1, int(stride))))
        cameras.append(dict(
            name=f"cam{c}",
            source=camera_source_label(cap.get('cam_topics'), cap.get('src_path'),
                                       cap.get('src_format'), c),
            cal_vec=np.asarray(cal.intr, float).tolist(),
            is_fisheye=bool(cal.is_fisheye), image_size=list(cam['image_size']),
            model=getattr(cal, "model", None) or ("equidistant" if cal.is_fisheye else "radtan"),
            T_CtoI=cal.T.matrix().tolist(), td=float(cal.td), readout=float(cal.tr),
            reproj_rms_px=float(e["reproj_rms_px"]),
            det=dict(t_cam=np.asarray(cam['t_cam'], float)[keep].tolist(),
                     uv_list=[np.asarray(cam['uv_list'][k], float).tolist() for k in keep],
                     ids_list=[np.asarray(cam['ids_list'][k], int).tolist() for k in keep])))
    imus = [dict(name="imu0", T_IstoI=np.eye(4).tolist())]       # base IMU at the body origin
    for m in (getattr(r, "imus", None) or []):
        T = m.get("T_IstoI")
        if T is not None:
            imus.append(dict(name=m.get("name", "aux"), T_IstoI=T.matrix().tolist()))
    return dict(t_states=ts.tolist(), T_states=T_states,         # T_ItoW (IMU world)
                T_BinW=r.T_BinW.matrix().tolist(),               # board at T_BtoW
                board_points=pts_B.tolist(),
                reproj_rms_px=float(cap['summary']['reproj_rms_px']),
                cameras=cameras, imus=imus)


def parse_rig_view(s, default=(20.0, -60.0)):
    """Parse a ``'elev,azim'`` string (degrees) into ``(elev, azim)`` for the rig panel's 3-D view
    angle; return ``default`` on empty / malformed / wrong-arity input."""
    if not s:
        return default
    try:
        e, a = s.split(",")
        return float(e), float(a)
    except (ValueError, AttributeError):
        return default


def _rig_extent(origins, pad=0.02):
    """Center + half-extent (m) of a set of sensor origins (K,3), padded — for a fixed, rig-scale
    cubic 3D box that frames the sensor cluster (NOT the meter-scale trajectory)."""
    P = np.asarray(origins, float).reshape(-1, 3)
    mn, mx = P.min(0), P.max(0)
    center = (mn + mx) / 2
    half = max(float((mx - mn).max()) / 2, 1e-6) + pad
    return center, half


def _draw_sensor_frame(ax, T, label, *, axis_len=0.03, kind="cam"):
    """Draw a sensor's coordinate frame at 4x4 world pose `T`: three short RGB axis lines from its
    origin + a name label. `kind` ('cam'|'imu') tints the label (blue cameras, red IMUs)."""
    T = np.asarray(T, float)
    o = T[:3, 3]
    for k, col in enumerate(("r", "g", "b")):                  # body x/y/z -> R/G/B
        a = o + axis_len * T[:3, k]
        ax.plot([o[0], a[0]], [o[1], a[1]], [o[2], a[2]], color=col, lw=1.5)
    ax.text(o[0], o[1], o[2], f"  {label}", fontsize=6,
            color="tab:blue" if kind == "cam" else "tab:red")


def _residual_color(err_px):
    """Color a per-corner reprojection residual by magnitude: green <0.5 px, orange <1 px,
    red ≥1 px — so high-error corners stand out in the detection-vs-reprojection panel."""
    return "tab:green" if err_px < 0.5 else ("tab:orange" if err_px < 1.0 else "tab:red")


def _scene_axes(ax, boards, board_poses, cam_path, *, frustum_at=None, upto=None):
    """Draw the static 3-D scene on `ax`: the (growing, if `upto` is given) camera path with
    start/end markers + each board's face and corners (per-board color) + an optional frustum at
    `(pos, R)`. Shared by both video modes; only the frustum/path-extent change per frame."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    cam_path = np.asarray(cam_path, float)
    _draw_path(ax, cam_path, upto=upto)
    allpts = [cam_path]
    for k, (board, T) in enumerate(zip(boards, board_poses)):
        col = _BOARD_COLORS[k % len(_BOARD_COLORS)]
        pB = np.asarray(board.all_points_3d(), float)
        pW = np.array([np.asarray(T.transformFrom(gtsam.Point3(*p)), float) for p in pB])
        ax.scatter(pW[:, 0], pW[:, 1], pW[:, 2], color=col, s=6)
        ax.add_collection3d(Poly3DCollection([_board_face(board, T)], alpha=0.22,
                                             facecolor=col, edgecolor="k"))
        allpts.append(pW)
    if frustum_at is not None:
        _draw_frustum(ax, frustum_at[0], frustum_at[1])
    _equal_aspect(ax, np.vstack(allpts))
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")


def _boards_of(scn):
    """(boards, board_poses) for either a single-board CalibScenario or an
    N-board MultiBoardScenario."""
    if hasattr(scn, "boards"):
        return list(scn.boards), list(scn.T_BinW_list)
    return [scn.board], [scn.T_BinW]


def _cam_pose(traj, T_CtoI, t):
    """T_CtoW at time t: T_ItoW(t)·T_CtoI (traj returns R_GtoI, p)."""
    R_GtoI, p = traj.get_pose(float(t))
    return gtsam.Pose3(gtsam.Rot3(R_GtoI.T), gtsam.Point3(*p)).compose(T_CtoI)


def _camera_track(traj, T_CtoI, times):
    """Camera position + rotation in world at each time."""
    pos, rot = [], []
    for t in times:
        T_CtoW = _cam_pose(traj, T_CtoI, t)
        pos.append(np.asarray(T_CtoW.translation(), float))
        rot.append(T_CtoW.rotation().matrix())
    return np.array(pos), rot


def rotation_curve_deg(rotations):
    """Continuous per-axis orientation curve (deg) from a dense sequence of R_ItoW
    matrices: cumulative body-frame increments Log(R_{i-1}ᵀ·R_i), starting at 0.
    Euler rpy + np.unwrap breaks at gimbal lock (pitch ±90° makes roll/yaw flip
    ~180° and unwrap ratchets the flips into fake revolutions); this stays smooth
    and bounded for oscillating motion."""
    th = np.zeros((len(rotations), 3))
    for i in range(1, len(rotations)):
        dR = gtsam.Rot3.ClosestTo(rotations[i - 1].T @ rotations[i])
        th[i] = th[i - 1] + gtsam.Rot3.Logmap(dR)
    return np.rad2deg(th)


def rotation_log_rad(rotations):
    """Orientation as the log(R) rotation vector (rad) relative to the first sample:
    Log(R_0ᵀ·R_i) for a sequence of R_ItoW matrices. No Euler angles, so no gimbal
    lock; the curve only jumps if the total rotation from start crosses π."""
    R0 = np.asarray(rotations[0], float)
    return np.array([gtsam.Rot3.Logmap(gtsam.Rot3.ClosestTo(R0.T @ np.asarray(R, float)))
                     for R in rotations])


def render_sim_video(source, out, *, fps=20, dpi=110, window=None,
                     image_panel=True, max_seconds=None) -> str:
    """Render a simulation scene: the rig flying its trajectory, fixed board(s), a
    moving frustum, and (optionally) the simulated camera image. `source` is a
    `SimConfig` (→ `.build()`) or an already-built scenario. Returns the path written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scn = source.build() if hasattr(source, "build") else source
    t0 = scn.traj.start_time if window is None else float(window[0])
    t1 = scn.traj.end_time if window is None else float(window[1])
    if max_seconds is not None:
        t1 = min(t1, t0 + float(max_seconds))
    times = np.arange(t0, t1, 1.0 / fps)
    cam_pos, cam_R = _camera_track(scn.traj, scn.T_CtoI, times)
    boards, board_poses = _boards_of(scn)
    w, h = scn.image_size

    fig = plt.figure(figsize=(13, 6) if image_panel else (7, 6))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d") if image_panel else fig.add_subplot(111, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2) if image_panel else None

    def frame(i):
        """Render animation frame `i`: the 3D scene with the frustum at that pose,
        plus (when `image_panel`) the corners the camera sees at that instant."""
        ax3.clear()
        _scene_axes(ax3, boards, board_poses, cam_pos,
                    frustum_at=(cam_pos[i], cam_R[i]))
        ax3.set_title(f"trajectory + boards   t={times[i] - t0:.2f}s")
        if ax2 is not None:
            ax2.clear()
            ax2.add_patch(plt.Rectangle((0, 0), w, h, fill=False, edgecolor="k"))
            cam = scn.camera([float(times[i])])
            uv, ids = cam.uv[0], cam.ids[0]
            if len(uv):
                ax2.scatter(uv[:, 0], uv[:, 1], c="tab:red", s=12)
            ax2.set_xlim(-20, w + 20); ax2.set_ylim(h + 20, -20)
            ax2.set_aspect("equal"); ax2.set_xlabel("u"); ax2.set_ylabel("v")
            ax2.set_title(f"camera view   ({len(ids)} corners)")

    return _animate(fig, frame, n_frames=len(times), fps=fps, out=out, dpi=dpi)


from dataclasses import dataclass


@dataclass
class CalibScene:
    """The scene a real calibration result describes, for `render_calib_video`:
    the recovered trajectory + board geometry/poses + the recovered camera."""
    traj: object                       # B-spline (recovered or GT trajectory)
    board_points: dict                 # {board_id: (N,3) board-frame corners}
    board_poses: dict                  # {board_id: gtsam.Pose3 in world}
    cal_vec: np.ndarray                # [fx,fy,cx,cy, dist...]
    is_fisheye: bool
    image_size: tuple
    T_CtoI: gtsam.Pose3
    td: float = 0.0                    # camera time offset: world pose at t_cam + td
    readout: float = 0.0               # rolling-shutter readout (s); per-row capture time
    rolling_mode: int = 0              # 0 top / 1 middle / 2 bottom (matches the cam factor)
    model: str | None = None           # distortion model; None = legacy is_fisheye
                                       # (radtan / equidistant, 8-long cal_vec only)


class _PointBoard:
    """Adapt a raw (N,3) corner array to the `.all_points_3d()` board interface
    that `_scene_axes` / `_board_face` expect."""
    def __init__(self, pts):
        self._p = np.asarray(pts, float)

    def all_points_3d(self):
        return self._p


def calib_scene_from_result(result, board_points_by_id, *, image_size,
                            fit_dt: float = 0.05) -> "CalibScene":
    """Build a `CalibScene` from a `CamImuCalibResult` — the RECOVERED trajectory
    (state poses fit to a B-spline), recovered board pose(s), and recovered camera
    (intrinsics + `T_CtoI`). Pair the returned scene with the ORIGINAL cam-dict
    (the calibration's input detections) when calling `render_calib_video`, e.g.::

        res = run_multi_board_calibration(cam, board_points, ...)
        scene = calib_scene_from_result(res, board_points, image_size=(640, 480))
        render_calib_video(scene, cam, "calib.mp4")
    """
    from mvis.calibration.pipeline_cam_imu import X        # lazy: avoids import cycle
    from mvis.simulation.from_result import fit_bspline_from_states

    t_states = np.asarray(result.t_states, float)
    T_states = [result.values.atPose3(X(k)).matrix() for k in range(len(t_states))]
    traj, _err = fit_bspline_from_states(t_states, T_states, fit_dt)
    board_poses = {int(b): gtsam.Pose3(np.asarray(M, float))
                   for b, M in result.diagnostics['board_poses'].items()}
    cal = result.calib
    return CalibScene(
        traj=traj,
        board_points={int(b): np.asarray(p, float) for b, p in board_points_by_id.items()},
        board_poses=board_poses, cal_vec=np.asarray(cal.intr, float),
        is_fisheye=bool(cal.is_fisheye), image_size=tuple(image_size), T_CtoI=cal.T,
        td=float(cal.td), readout=float(cal.tr), model=str(cal.model))


def _reproject_frame(scene: CalibScene, t: float, ids, rows=None) -> np.ndarray:
    """Reproject the detected ids' board points through the recovered calibration at
    frame time `t`. Uses the SAME projector + rolling-shutter row map as the GTSAM
    calibration factor (`factors.projection.make_projector`, `shutter.row_offset`), so the
    overlay is automatically consistent with the solver. Each corner is projected at
    its own capture time `t + td + ρ(row)·readout` (ρ from the measured row = `rows`,
    i.e. the detected `v`); pass `rows=None` for global-shutter. Returns (N,2) uv
    (NaN for behind-camera / out-of-model-range)."""
    from mvis.calibration.factors.projection import make_projector
    from mvis.core.shutter import row_offset
    from mvis.simulation.multi_board import board_of, local_of

    ids = np.asarray(ids).ravel()
    if not scene.board_points:          # no board geometry -> can't reproject (detection-only)
        return np.full((len(ids), 2), np.nan)
    cal_vec = np.asarray(scene.cal_vec, float)
    proj = make_projector(scene.model or legacy_model(scene.is_fisheye, cal_vec), cal_vec)
    h = int(scene.image_size[1])
    rs = (rows is not None) and (scene.readout != 0.0)
    rows = np.asarray(rows, float).ravel() if rows is not None else None
    uv = []
    for n, gid in enumerate(ids):
        b, loc = int(board_of(gid)), int(local_of(gid))
        pB = np.asarray(scene.board_points[b], float)[loc]
        pW = scene.board_poses[b].transformFrom(gtsam.Point3(*pB))
        rho = row_offset(float(rows[n]), h, scene.rolling_mode) if rs else 0.0
        T_CtoW = _cam_pose(scene.traj, scene.T_CtoI, t + scene.td + rho * scene.readout)
        try:
            px, _, _, _ = proj.project(T_CtoW, gtsam.Point3(*np.asarray(pW, float)), jac=False)
            uv.append(np.asarray(px, float))
        except RuntimeError:            # cheirality / out of model range
            uv.append([np.nan, np.nan])
    return np.array(uv, float).reshape(-1, 2)


def _trace_panels(axp, axr, t, P, A):
    """The two trace panels under a video: IMU-world position (m) and orientation as log(R) from the
    start (rad) against time. Legends sit above the axes in one row, so they never hide a curve."""
    for ax, Y, labels, ylabel, title in (
            (axp, P, list("xyz"), "position (m)", "IMU world trajectory"),
            (axr, A, [fr"$\log(R)_{c}$" for c in "xyz"], "log(R) from start (rad)", "IMU world orientation")):
        for i, lab in enumerate(labels):
            ax.plot(t, Y[:, i], label=lab)
        ax.set_ylabel(ylabel); ax.set_xlabel("time (s)"); ax.grid(alpha=0.3)
        ax.set_title(title, loc="left", fontsize=10)
        ax.legend(fontsize=8, ncol=3, loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False,
                  borderaxespad=0.2)


def render_calib_video(scene: CalibScene, detections, out, *, fps=20, dpi=110,
                       max_frames=None, show_trajectory=False) -> str:
    """Render a real calibration result: the recovered trajectory + board(s) + a
    moving frustum, with the image panel PLAYING THE PER-FRAME DETECTIONS — the
    detected corners, the reprojected corners, the residual segments, and the
    per-frame reproj RMS. `detections` is a cam-dict (`t_cam`, `uv_list`, `ids_list`).
    Returns the path written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_cam = np.asarray(detections['t_cam'], float)
    idx = [k for k in range(len(t_cam)) if len(detections['ids_list'][k])]
    if max_frames is not None:
        idx = idx[:int(max_frames)]
    times = t_cam[idx]
    cam_pos, cam_R = _camera_track(scene.traj, scene.T_CtoI, times)
    bids = sorted(scene.board_points) if scene.board_points else []  # empty -> no 3-D board / no reproj
    boards = [_PointBoard(scene.board_points[b]) for b in bids]
    board_poses = [scene.board_poses[b] for b in bids]
    w, h = scene.image_size

    cursors = []
    if show_trajectory:
        fig = plt.figure(figsize=(13, 9))
        ax3 = fig.add_subplot(2, 2, 1, projection="3d")
        ax2 = fig.add_subplot(2, 2, 2)
        axp = fig.add_subplot(2, 2, 3)
        axr = fig.add_subplot(2, 2, 4)
        # static trajectory panels: IMU-world position + orientation over the window
        tg = np.linspace(times[0], times[-1], 400)
        P, R = [], []
        for t in tg:
            R_GtoI, p = scene.traj.get_pose(float(t))
            P.append(np.asarray(p, float))
            R.append(R_GtoI.T)
        P = np.array(P); A = rotation_log_rad(R)
        _trace_panels(axp, axr, tg - tg[0], P, A)
    else:
        fig = plt.figure(figsize=(13, 6))
        ax3 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2)
        axp = axr = None

    def frame(j):
        """Render animation frame `j` (source index `idx[j]`): the 3D scene, the
        detections panel, and — when `show_trajectory` — the time cursor tracking
        the position/rotation curves."""
        k = idx[j]
        if show_trajectory:                       # move the time cursor on the curves
            while cursors:
                cursors.pop().remove()
            tc = times[j] - times[0]
            cursors.append(axp.axvline(tc, color="r", lw=1.2))
            cursors.append(axr.axvline(tc, color="r", lw=1.2))
        ax3.clear()
        _scene_axes(ax3, boards, board_poses, cam_pos, frustum_at=(cam_pos[j], cam_R[j]))
        ax3.set_title(f"recovered trajectory + board(s)   t={times[j] - times[0]:.2f}s")
        ax2.clear()
        ax2.add_patch(plt.Rectangle((0, 0), w, h, fill=False, edgecolor="k"))
        det = np.asarray(detections['uv_list'][k], float).reshape(-1, 2)
        ids = np.asarray(detections['ids_list'][k])
        rep = _reproject_frame(scene, float(t_cam[k]), ids, rows=det[:, 1])
        ax2.scatter(det[:, 0], det[:, 1], facecolors="none", edgecolors="tab:green",
                    s=26, linewidths=1.0, label="detected")
        has_rep = np.isfinite(rep).any()
        if has_rep:                                       # reprojection overlay (needs board geometry)
            for d, r in zip(det, rep):
                if np.all(np.isfinite(r)):
                    ax2.plot([d[0], r[0]], [d[1], r[1]], color="0.5", lw=0.5)
            ax2.scatter(rep[:, 0], rep[:, 1], c="tab:red", s=9, label="reprojected")
            res = np.linalg.norm(det - rep, axis=1)
            rms = float(np.sqrt(np.nanmean(res ** 2))) if np.isfinite(res).any() else float("nan")
            title = f"detections vs reprojection   RMS={rms:.2f} px   ({len(ids)} corners)"
        else:
            title = f"detections   ({len(ids)} corners)"
        ax2.set_xlim(-20, w + 20); ax2.set_ylim(h + 20, -20); ax2.set_aspect("equal")
        ax2.set_xlabel("u"); ax2.set_ylabel("v")
        ax2.legend(loc="upper right", fontsize=7)
        ax2.set_title(title)

    return _animate(fig, frame, n_frames=len(idx), fps=fps, out=out, dpi=dpi)


def render_detections_video(detections, image_size, out, *, fps=20, dpi=110,
                            max_frames=None, stride=1, title="detections") -> str:
    """Single-panel video that plays per-frame detected corners (no 3-D scene / no
    reprojection) — for raw detection data (e.g. a real DetectionCache) where no
    trajectory/board pose is available. `detections` is a cam-dict. Returns the path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.asarray(detections['t_cam'], float)
    idx = [k for k in range(0, len(t), max(1, int(stride))) if len(detections['ids_list'][k])]
    if max_frames is not None:
        idx = idx[:int(max_frames)]
    w, h = image_size
    fig, ax = plt.subplots(figsize=(7.2, 7.2 * h / w))
    t0 = t[idx[0]] if idx else 0.0

    def frame(j):
        """Render animation frame `j`: the detected corners for source frame
        `idx[j]`, drawn in image coordinates inside the image rectangle."""
        k = idx[j]
        ax.clear()
        ax.add_patch(plt.Rectangle((0, 0), w, h, fill=False, edgecolor="k"))
        uv = np.asarray(detections['uv_list'][k], float).reshape(-1, 2)
        ax.scatter(uv[:, 0], uv[:, 1], c="tab:green", s=10)
        ax.set_xlim(-20, w + 20); ax.set_ylim(h + 20, -20); ax.set_aspect("equal")
        ax.set_xlabel("u"); ax.set_ylabel("v")
        ax.set_title(f"{title}   t={t[k] - t0:.2f}s   ({len(uv)} corners)")

    return _animate(fig, frame, n_frames=len(idx), fps=fps, out=out, dpi=dpi)


def _detections_from_cache(path, cam_index=0) -> dict:
    """Read a DetectionCache json (mvis.io.detection_cache format) into a cam-dict
    (`t_cam`, `uv_list`, `ids_list`, `image_size`) for one camera."""
    import json
    with open(path) as f:
        data = json.load(f)
    cam = data["cameras"][str(cam_index)]
    t_cam, uv_list, ids_list, img = [], [], [], None
    for fr in cam["frames"]:
        obs = fr["observations"]
        t_cam.append(float(fr["timestamp"]))
        uv_list.append(np.array([o["pixel"] for o in obs], float).reshape(-1, 2))
        ids_list.append(np.array([o["point_id"] for o in obs], int))
        if fr.get("image_size"):
            img = tuple(fr["image_size"])
    return dict(t_cam=np.array(t_cam, float), uv_list=uv_list, ids_list=ids_list,
                image_size=img)


def board_points_from(spec) -> dict:
    """Build `board_points_by_id` ({0: (N,3) board-frame corners, indexed by
    point_id}) for the reprojection overlay. `spec` may be a `Board` instance, a
    Kalibr/AprilGrid target YAML path, or a `BoardConfig`."""
    from mvis.detection.board import Board
    if isinstance(spec, Board):
        board = spec
    elif isinstance(spec, str):
        from mvis.detection.apriltag_board import AprilTagBoard
        board = AprilTagBoard.from_kalibr_yaml(spec)
    else:
        from mvis.detection.board_factory import create_board
        board = create_board(spec)
    return {0: np.asarray(board.all_points_3d(), float)}


def render_calib_video_from_path(result_path, out, *, detections_path,
                                 board_points=None, board_spec=None, cam_index=0,
                                 fps=20, dpi=110, max_frames=None) -> str:
    """Render a real calibration straight from FILES on disk: a GroundTruth/result
    json (`result_path`, via `gt_from_json` → trajectory + board pose + camera) and a
    DetectionCache json (`detections_path` → per-frame detections). `board_points`
    (the board model, not stored on disk) enables the reprojection overlay; pass it
    directly or as `board_spec` (a Board / Kalibr YAML path / BoardConfig). Without a
    board model the panel still plays the detections frame by frame. Returns the path
    written."""
    from mvis.simulation.from_result import (
        gt_from_json, fit_bspline_from_states, gt_cam_model)

    if board_points is None and board_spec is not None:
        board_points = board_points_from(board_spec)
    gt = gt_from_json(result_path)
    traj, _err = fit_bspline_from_states(
        np.asarray(gt.t_states, float),
        [np.asarray(T, float) for T in gt.T_states], 0.05)
    cam = gt.cams[cam_index]
    scene = CalibScene(
        traj=traj, board_points=board_points,
        board_poses={0: gtsam.Pose3(np.asarray(gt.T_BinW, float))},
        cal_vec=np.asarray(cam['intr8'], float), is_fisheye=bool(cam['is_fisheye']),
        image_size=tuple(cam['image_size']), T_CtoI=gtsam.Pose3(np.asarray(cam['T_CtoI'], float)),
        model=gt_cam_model(cam))
    det = _detections_from_cache(detections_path, cam_index)
    if det.get('image_size'):
        scene.image_size = det['image_size']
    return render_calib_video(scene, det, out, fps=fps, dpi=dpi, max_frames=max_frames)


def render_calib_video_multicam(cameras, out, *, fps=20, dpi=110, stride=1,
                                max_frames=None, show_trajectory=True,
                                det_label="detected", imus=None, rig_view=(20.0, -60.0),
                                rs_compare=False) -> str:
    """Multi-camera world-frame video: the shared 3D trajectory + board, ALL cameras'
    detection-vs-reprojection panels (each with its own intrinsics/T_CtoI/td), and the
    position/orientation trajectory panels with a moving cursor.

    `rs_compare`: for rolling-shutter cameras (readout != 0) also overlay the SAME
    calibration reprojected with the readout term switched off (every row at the
    frame time) — the global-shutter-model error the readout estimate removes.

    `cameras`: list of {name, scene: CalibScene, det: cam-dict}. All scenes share the
    same traj + board_points/board_poses; each carries its own camera + td. The master
    time grid is the first camera's frames; each panel shows its nearest frame in time.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    N = len(cameras)
    s0 = cameras[0]['scene']
    traj = s0.traj
    bt = np.asarray(cameras[0]['det']['t_cam'], float)
    midx = [k for k in range(0, len(bt), max(1, int(stride)))
            if traj.start_time <= bt[k] <= traj.end_time and len(cameras[0]['det']['ids_list'][k])]
    if max_frames is not None:
        midx = midx[:int(max_frames)]
    times = bt[midx]
    cam_t = [np.asarray(c['det']['t_cam'], float) for c in cameras]
    # Per-camera "is this frame current" gate: half its own frame period (floored
    # at the legacy 60 ms). A fixed 60 ms blanks slow cameras — e.g. a 4.3 Hz cam
    # (233 ms period) on a 20 Hz master clock shows nothing 83% of the time; with
    # the adaptive gate its panel holds the most recent frame instead.
    cam_tol = [max(0.06, 0.55 * float(np.median(np.diff(ct)))) if len(ct) > 1 else 0.06
               for ct in cam_t]
    boards = [_PointBoard(s0.board_points[b]) for b in sorted(s0.board_points)] if s0.board_points else []
    bposes = [s0.board_poses[b] for b in sorted(s0.board_poses)] if s0.board_points else []
    board_total = sum(len(s0.board_points[b]) for b in s0.board_points) if s0.board_points else 0
    # per-camera world tracks (cameras are rigid to the IMU) for arrows + base frustum
    cam_tracks = [_camera_track(traj, c['scene'].T_CtoI, times) for c in cameras]
    cam_pos = cam_tracks[0][0]                                  # base path for the trajectory line

    # --- rig sensor-layout panel: every camera + IMU as a body frame, zoomed + moving ---
    rig, rig_half = [], 0.0                                     # [(name, kind, world_track)]
    if imus:
        for c in cameras:
            rig.append((c['name'].replace(' (sim)', ''), 'cam',
                        _camera_track(traj, c['scene'].T_CtoI, times)))
        for m in imus:
            rig.append((m['name'], 'imu', _camera_track(traj, m['T_IstoI'], times)))
        body = ([np.asarray(c['scene'].T_CtoI.translation(), float) for c in cameras]
                + [np.asarray(m['T_IstoI'].translation(), float) for m in imus])
        _, rig_half = _rig_extent(body, pad=0.03)

    npanel = N + (1 if imus else 0)                            # camera panels + the rig panel
    ncr = int(np.ceil(npanel / 2))
    nrows = ncr + (1 if show_trajectory else 0)
    fig = plt.figure(figsize=(16, 4.2 * nrows))
    gs = GridSpec(nrows, 4, figure=fig)
    ax3 = fig.add_subplot(gs[0:ncr, 0:2], projection="3d")
    caxes = [fig.add_subplot(gs[i // 2, 2 + i % 2]) for i in range(N)]
    ax_rig = fig.add_subplot(gs[N // 2, 2 + N % 2], projection="3d") if imus else None
    cursors = []
    if show_trajectory:
        bottom = gs[ncr, :].subgridspec(1, 2, wspace=0.28)   # room for the right panel's y-label
        axp, axr = fig.add_subplot(bottom[0]), fig.add_subplot(bottom[1])
        tg = np.linspace(times[0], times[-1], 400)
        P = np.array([np.asarray(traj.get_pose(float(t))[1], float) for t in tg])
        A = rotation_log_rad([traj.get_pose(float(t))[0].T for t in tg])
        _trace_panels(axp, axr, tg - tg[0], P, A)

    def frame(j):
        """Render animation frame `j` of the multi-camera view: the scene with the
        path grown up to `j` and one colored optical-axis arrow per camera."""
        t = times[j]
        ax3.clear()
        _scene_axes(ax3, boards, bposes, cam_pos, upto=j)      # GROWING path + start/end markers
        for ci, (cp, cR) in enumerate(cam_tracks):             # z-arrow per camera
            col = _CAM_COLORS[ci % len(_CAM_COLORS)]
            _draw_optical_axis(ax3, cp[j], cR[j], color=col,
                               lw=2.4 if ci == 0 else 1.6, length=0.5)
        _draw_frustum(ax3, cam_tracks[0][0][j], cam_tracks[0][1][j],   # base: small red frustum
                      color="red", lw=1.6, scale=0.05)
        ax3.legend([plt.Line2D([0], [0], color=_CAM_COLORS[ci % len(_CAM_COLORS)], lw=2)
                    for ci in range(len(cameras))],
                   [c['name'] for c in cameras], fontsize=6, loc="upper left")
        ax3.set_title(f"trajectory + cameras   t={t - times[0]:.1f}s   "
                      f"(frame {j + 1}/{len(midx)})")
        for cax, cam, ct, tol in zip(caxes, cameras, cam_t, cam_tol):
            cax.clear()
            scn = cam['scene']; w, h = scn.image_size
            src = cam.get('source') or cam['name']            # rostopic / euroc folder / cam<c>
            cax.add_patch(plt.Rectangle((0, 0), w, h, fill=False, edgecolor="k"))
            k = int(np.argmin(np.abs(ct - t)))
            if abs(ct[k] - t) < tol and len(cam['det']['ids_list'][k]):
                det = np.asarray(cam['det']['uv_list'][k], float).reshape(-1, 2)
                ids = np.asarray(cam['det']['ids_list'][k])
                rep = _reproject_frame(scn, float(ct[k]), ids, rows=det[:, 1])
                cax.scatter(det[:, 0], det[:, 1], facecolors="none",
                            edgecolors="tab:green", s=12, linewidths=0.8, label=det_label)
                if np.isfinite(rep).any():
                    res = np.linalg.norm(det - rep, axis=1)
                    for d, r, e in zip(det, rep, res):            # residual segment, colored by |err|
                        if np.all(np.isfinite(r)):
                            cax.plot([d[0], r[0]], [d[1], r[1]], color=_residual_color(e), lw=0.9)
                    cax.scatter(rep[:, 0], rep[:, 1], c="tab:red", s=5, label="reprojected")
                    rms = float(np.sqrt(np.nanmean(res ** 2)))
                    cov = f"{len(ids)}/{board_total}" if board_total else f"{len(ids)}"
                    title = f"{src}\nRMS={rms:.2f}px  ({cov} corners)"
                    if rs_compare and scn.readout != 0.0:
                        gs = _reproject_frame(scn, float(ct[k]), ids, rows=None)
                        cax.scatter(gs[:, 0], gs[:, 1], marker="x", c="tab:blue", s=10,
                                    linewidths=0.7, label="global-shutter model")
                        rms_gs = float(np.sqrt(np.nanmean(np.sum((det - gs) ** 2, axis=1))))
                        title = (f"{src}  ({cov} corners)\nRMS rolling {rms:.2f}px | "
                                 f"global-shutter model {rms_gs:.2f}px  (tr {scn.readout * 1e3:.1f} ms)")
                else:
                    cov = f"{len(ids)}/{board_total}" if board_total else f"{len(ids)}"
                    title = f"{src}\n({cov} corners)"
                cax.legend(loc="upper right", fontsize=5, framealpha=0.7)
            else:
                title = f"{src}\n(no detection)"
            cax.set_xlim(-20, w + 20); cax.set_ylim(h + 20, -20)
            cax.set_aspect("equal"); cax.set_title(title, fontsize=8)
            cax.tick_params(labelsize=6)
        if ax_rig is not None:                            # rig: sensors as body frames, zoomed + moving
            ax_rig.clear()
            cen = np.mean([trk[0][j] for _, _, trk in rig], axis=0)   # world centroid of the rig
            for nm, kind, (pos, R) in rig:
                T = np.eye(4); T[:3, :3] = R[j]; T[:3, 3] = pos[j]
                _draw_sensor_frame(ax_rig, T, nm, kind=kind, axis_len=rig_half * 0.3)
            ax_rig.set_xlim(cen[0] - rig_half, cen[0] + rig_half)
            ax_rig.set_ylim(cen[1] - rig_half, cen[1] + rig_half)
            ax_rig.set_zlim(cen[2] - rig_half, cen[2] + rig_half)
            ax_rig.view_init(elev=rig_view[0], azim=rig_view[1])   # adjustable; rig tumbles in place
            ax_rig.set_title("rig: cameras (blue) + IMUs (red)", fontsize=8)
            ax_rig.tick_params(labelsize=5)
        if show_trajectory:
            while cursors:
                cursors.pop().remove()
            tc = t - times[0]
            cursors.append(axp.axvline(tc, color="r", lw=1.2))
            cursors.append(axr.axvline(tc, color="r", lw=1.2))

    return _animate(fig, frame, n_frames=len(midx), fps=fps, out=out, dpi=dpi)


def _animate(fig, frame_fn, *, n_frames, fps, out, dpi=110) -> str:
    """Save a `FuncAnimation(fig, frame_fn)` of `n_frames` to `out`. Prefers an MP4
    (ffmpeg) when `out` ends in .mp4 and ffmpeg is available; otherwise writes a GIF
    (Pillow), swapping the extension. Returns the path actually written."""
    from matplotlib import animation

    anim = animation.FuncAnimation(fig, frame_fn, frames=int(n_frames), blit=False)
    want_mp4 = str(out).lower().endswith(".mp4")
    if want_mp4 and animation.FFMpegWriter.isAvailable():
        writer = animation.FFMpegWriter(fps=int(fps))
        path = out
    else:
        if want_mp4:                       # ffmpeg missing -> GIF fallback
            out = os.path.splitext(out)[0] + ".gif"
        writer = animation.PillowWriter(fps=int(fps))
        path = out
    anim.save(path, writer=writer, dpi=dpi)
    return path
